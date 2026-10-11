using System.Collections.Generic;
using BurstPQS.Collections;
using BurstPQS.Util;
using Unity.Burst;
using Unity.Collections;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Rendering;
using Stopwatch = System.Diagnostics.Stopwatch;

namespace BurstPQS;

// Draws each complete family of four sibling leaf quads as one mesh.
//
// Every leaf quad has its own renderer, so terrain costs one draw per quad in
// every pass that draws it (depth, forward, shadows, reflection probe). When
// all four children of a quad are visible leaves that would draw the same way,
// their meshes are copied into one mesh on a renderer under the parent quad
// and the children stop rendering through forceRenderingOff.
//
// Nothing else about the quads changes. They keep their meshes and colliders,
// and meshRenderer.enabled keeps meaning "this quad is visible" for stock and
// for mods.
//
// The family mesh has the same vertex layout as its quads, with each quad in
// its own slice. A family forms as soon as its parent subdivides, so each
// child's build is written into its slice as it completes, and every later
// write BatchPQS makes to a quad's mesh is made to its slice too. Only quads
// that were built before their family formed (after a collapse, or when cached
// children come back) are read back from their meshes.
[BurstCompile]
internal sealed class QuadMerger(BatchPQS batchPQS)
{
    internal static bool Enabled = true;

    // Families are checked when one of their quads changes, and mods report
    // renderer changes through BatchPQS.OnQuadRendererChanged. As a backstop
    // for changes nothing reports, this many families are also rechecked each
    // frame.
    const int ChecksPerFrame = 1;

    // A family that failed the renderer checks is retried after this many frames.
    const int RejectFrames = 60;

    // Main thread time per frame for reading back quads. Families still
    // waiting draw their children until they are copied.
    static readonly long CopyBudget = Stopwatch.Frequency / 4000;

    // Unused family objects kept for reuse, per sphere.
    const int MaxPooled = 256;

    // Accumulated cost, for profiling.
    internal static long RecheckTicks;
    internal static long ChangeTicks;
    internal static long CopyTicks;
    internal static long MirrorTicks;
    internal static int CopiedQuads;
    internal static int MirroredWrites;
    internal static int Frames;
    static int lastFrame = -1;

    static readonly ProfilerMarker ReconcileMarker = new("QuadMerger.Reconcile");

    // The objects that draw a family. Pooled and reused across families.
    sealed class Shell
    {
        public GameObject gameObject;
        public MeshRenderer renderer;
        public Mesh mesh;

        // The vertex layout the mesh was created with, or null.
        public VertexAttributeDescriptor[] attributes;

        // Floats per vertex in each stream, and each attribute's offset in
        // floats within its stream.
        public readonly int[] strides = new int[4];
        public readonly int[] offsets = new int[VertexAttributeCount];

        // How each stream is read back, and its attribute if it has only one.
        public readonly StreamKind[] kinds = new StreamKind[4];
        public readonly VertexAttributeDescriptor[] single = new VertexAttributeDescriptor[4];
    }

    enum StreamKind
    {
        None,

        // One attribute, uploaded straight from its getter.
        Single,

        // BatchPQS's interleaved stream, interleaved by Burst.
        Interleaved,

        // Anything else, interleaved as floats.
        Floats,
    }

    sealed class Group
    {
        public PQ parent;
        public readonly PQ[] children = new PQ[4];
        public readonly Bounds[] bounds = new Bounds[4];
        public Shell shell;

        // Children whose slice is out of date and must be read back.
        public int stale = 0xF;

        // Whether all four children are built, visible and render alike. Until
        // then the family is pending and its children draw themselves.
        public bool ready;

        // Whether the family mesh is drawn instead of the children.
        public bool merged;

        // Set when a child could not be copied, until a child changes again.
        public bool failed;

        // Whether the group is in copyQueue.
        public bool queued;

        // Set once dissolved, for references still held in copyQueue.
        public bool removed;

        // Position in groupList.
        public int index;
    }

    const int VertexAttributeCount = (int)VertexAttribute.BlendIndices + 1;

    readonly Dictionary<PQ, Group> groups = [];
    readonly List<Group> groupList = [];
    readonly Stack<Shell> pool = [];
    readonly HashSet<PQ> rejected = [];
    readonly HashSet<PQ> changed = [];
    readonly List<Group> copyQueue = [];
    readonly List<Group> copyRequeue = [];
    bool scanAll = true;
    int frame;
    int checkCursor;

    static readonly List<Material> MaterialsA = [];
    static readonly List<Material> MaterialsB = [];
    static readonly VertexAttributeDescriptor[] AttributeScratch = new VertexAttributeDescriptor[
        VertexAttributeCount
    ];
    static readonly List<Vector2> Vector2Scratch = [];
    static readonly List<Vector2> Uv2Scratch = [];
    static readonly List<Vector2> Uv3Scratch = [];
    static readonly List<Vector3> Vector3Scratch = [];
    static readonly List<Vector4> Vector4Scratch = [];
    static readonly List<Color> ColorScratch = [];
    static readonly List<int> IndexScratch = [];
    static readonly float[][] StreamScratch =
    [
        [],
        [],
        [],
        [],
    ];
    static ushort[] IndexSlice = [];

    const MeshUpdateFlags UpdateFlags =
        MeshUpdateFlags.DontValidateIndices
        | MeshUpdateFlags.DontRecalculateBounds
        | MeshUpdateFlags.DontResetBoneBounds;

    static int VertsPerQuad => PQS.cacheVertCount;
    static int IndicesPerQuad => PQS.cacheIndices[0].Length;

    #region Change tracking
    // A quad's mesh changed in a way that was not mirrored, so its slice must
    // be read back.
    internal void MarkDirty(PQ quad)
    {
        if (!TryGetSlice(quad, out var group, out int index))
            return;

        group.stale |= 1 << index;
        group.failed = false;
        Queue(group);
    }

    // A quad was subdivided, collapsed, built or shown or hidden, which can
    // make or break its own family and its parent's.
    internal void MarkChanged(PQ quad)
    {
        changed.Add(quad);
        var parent = quad.parent;
        if (parent is not null)
            changed.Add(parent);
    }

    // A mod changed a quad's materials, mesh or renderer settings. Its family is
    // taken apart, and formed again from the new state if it still matches.
    internal void MarkRendererChanged(PQ quad)
    {
        var parent = quad.parent;
        if (parent is not null && groups.TryGetValue(parent, out var group))
            RemoveAt(group.index);
        MarkChanged(quad);
    }

    bool TryGetSlice(PQ quad, out Group group, out int index)
    {
        index = -1;
        var parent = quad.parent;
        if (parent is null || !groups.TryGetValue(parent, out group))
        {
            group = null;
            return false;
        }

        for (int i = 0; i < 4; ++i)
        {
            if (ReferenceEquals(group.children[i], quad))
            {
                index = i;
                return true;
            }
        }
        return false;
    }

    // Whether writes to this quad's mesh can be mirrored into its slice. A
    // stale slice is read back in full later instead.
    bool TryGetMirror(PQ quad, out Group group, out int index) =>
        TryGetSlice(quad, out group, out index)
        && (group.stale & (1 << index)) == 0
        && group.shell.attributes is not null;
    #endregion

    #region Mirroring
    // PendingBuild.Complete uploaded a freshly built quad.
    internal void MirrorBuild(
        PQ quad,
        VertexAttributeDescriptor[] attributes,
        NativeArray<InterleavedVertex> interleaved,
        NativeArray<Vector3> normals,
        NativeArray<Vector4> tangents,
        NativeArray<Vector2> uv1s,
        int[] indices,
        Bounds bounds
    )
    {
        if (!TryGetSlice(quad, out var group, out int index))
            return;

        // A build writes every stream, so it also fills a stale slice.
        var shell = group.shell;
        if (shell.attributes is null && group.stale == 0xF)
            CreateMesh(shell, attributes, attributes.Length);
        if (!SameLayout(shell.attributes, attributes, attributes.Length))
        {
            MarkDirty(quad);
            return;
        }

        long start = Stopwatch.GetTimestamp();
        var mesh = shell.mesh;
        int vertexCount = VertsPerQuad;
        int baseVertex = index * vertexCount;
        var offset = SliceOffset(group, quad);

        var shifted = new NativeArray<InterleavedVertex>(
            vertexCount,
            Allocator.Temp,
            NativeArrayOptions.UninitializedMemory
        );
        OffsetPositionsFunc ??= BurstUtil.MaybeCompileDelegate<OffsetPositionsDelegate>(
            OffsetPositions
        );
        OffsetPositionsFunc(interleaved, shifted, offset);
        mesh.SetVertexBufferData(shifted, 0, baseVertex, vertexCount, 0, UpdateFlags);
        shifted.Dispose();

        mesh.SetVertexBufferData(normals, 0, baseVertex, vertexCount, 1, UpdateFlags);
        int uv1Stream = 2;
        if (tangents.IsCreated)
        {
            mesh.SetVertexBufferData(tangents, 0, baseVertex, vertexCount, 2, UpdateFlags);
            uv1Stream = 3;
        }
        mesh.SetVertexBufferData(uv1s, 0, baseVertex, vertexCount, uv1Stream, UpdateFlags);
        WriteIndices(mesh, indices, index);

        bounds.center += offset;
        group.bounds[index] = bounds;
        UpdateBounds(group);
        group.stale &= ~(1 << index);
        Queue(group);

        MirroredWrites++;
        MirrorTicks += Stopwatch.GetTimestamp() - start;
    }

    // BatchPQS wrote one of the quad's single-attribute streams in place.
    internal void MirrorStream<T>(PQ quad, T[] data, int stream)
        where T : struct
    {
        if (!TryGetMirror(quad, out var group, out int index))
            return;

        long start = Stopwatch.GetTimestamp();
        int vertexCount = VertsPerQuad;
        group.shell.mesh.SetVertexBufferData(
            data,
            0,
            index * vertexCount,
            vertexCount,
            stream,
            UpdateFlags
        );
        MirroredWrites++;
        MirrorTicks += Stopwatch.GetTimestamp() - start;
    }

    // PQ.UpdateVisibility changed the quad's edge stitching.
    internal void MirrorIndices(PQ quad, int[] indices)
    {
        if (!TryGetMirror(quad, out var group, out int index))
            return;

        if (indices.Length != IndicesPerQuad)
        {
            MarkDirty(quad);
            return;
        }

        long start = Stopwatch.GetTimestamp();
        WriteIndices(group.shell.mesh, indices, index);
        MirroredWrites++;
        MirrorTicks += Stopwatch.GetTimestamp() - start;
    }

    delegate void OffsetPositionsDelegate(
        in NativeArray<InterleavedVertex> source,
        in NativeArray<InterleavedVertex> destination,
        in Vector3 offset
    );

    static OffsetPositionsDelegate OffsetPositionsFunc;

    [BurstCompile(FloatMode = FloatMode.Fast)]
    static void OffsetPositions(
        in NativeArray<InterleavedVertex> source,
        in NativeArray<InterleavedVertex> destination,
        in Vector3 offset
    )
    {
        var output = destination;
        for (int i = 0; i < source.Length; ++i)
        {
            var vertex = source[i];
            vertex.position += offset;
            output[i] = vertex;
        }
    }
    #endregion

    #region Reconcile
    // Runs once a frame, after the quad tree has been updated.
    internal void Reconcile(List<PQ> activeQuads)
    {
        if (!Enabled || !batchPQS.CanMergeQuads)
        {
            if (groupList.Count != 0 || pool.Count != 0)
                Clear();
            changed.Clear();
            scanAll = true;
            return;
        }

        using var scope = ReconcileMarker.Auto();
        long start = Stopwatch.GetTimestamp();

        frame++;
        if (Time.frameCount != lastFrame)
        {
            lastFrame = Time.frameCount;
            Frames++;
        }
        if (frame % RejectFrames == 0)
        {
            changed.UnionWith(rejected);
            rejected.Clear();
        }
        if (scanAll)
        {
            scanAll = false;
            foreach (var quad in activeQuads)
            {
                if (quad.IsNotNullOrDestroyed() && quad.isSubdivided)
                    changed.Add(quad);
            }
        }

        // A family that no longer matches is dissolved, and formed again below
        // if its children still render alike.
        int checks = Mathf.Min(ChecksPerFrame, groupList.Count);
        for (int n = 0; n < checks && groupList.Count != 0; ++n)
        {
            checkCursor %= groupList.Count;
            var group = groupList[checkCursor];
            if (IsIntact(group) && UpdateReady(group))
            {
                checkCursor++;
                continue;
            }

            RemoveAt(checkCursor);
            changed.Add(group.parent);
        }

        long changeStart = Stopwatch.GetTimestamp();
        RecheckTicks += changeStart - start;

        foreach (var quad in changed)
        {
            if (groups.TryGetValue(quad, out var group))
            {
                if (IsIntact(group) && UpdateReady(group))
                    continue;
                RemoveAt(group.index);
            }

            if (rejected.Contains(quad) || !IsLeafFamily(quad))
                continue;

            // Families waiting on builds form now, so the builds fill them.
            if (IsFamily(quad))
            {
                if (DrawsAlike(quad))
                    Add(quad).ready = true;
                else
                    rejected.Add(quad);
            }
            else if (HasUnbuiltChild(quad))
                Add(quad);
        }
        changed.Clear();

        long copyStart = Stopwatch.GetTimestamp();
        ChangeTicks += copyStart - changeStart;

        long deadline = copyStart + CopyBudget;
        foreach (var group in copyQueue)
        {
            group.queued = false;
            if (group.failed || group.removed || !group.ready)
                continue;

            if (group.stale != 0)
            {
                if (Stopwatch.GetTimestamp() > deadline)
                {
                    SetMerged(group, false);
                    copyRequeue.Add(group);
                    continue;
                }

                if (!Copy(group))
                {
                    group.failed = true;
                    SetMerged(group, false);
                    continue;
                }
            }

            SetMerged(group, true);
        }
        copyQueue.Clear();
        foreach (var group in copyRequeue)
            Queue(group);
        copyRequeue.Clear();

        CopyTicks += Stopwatch.GetTimestamp() - copyStart;
    }

    void Queue(Group group)
    {
        if (group.queued)
            return;
        group.queued = true;
        copyQueue.Add(group);
    }

    internal void Clear()
    {
        foreach (var group in groupList)
            Dissolve(group);
        groupList.Clear();
        groups.Clear();
        copyQueue.Clear();

        while (pool.Count != 0)
        {
            var shell = pool.Pop();
            if (shell.gameObject.IsNotNullOrDestroyed())
                Object.Destroy(shell.gameObject);
            if (shell.mesh.IsNotNullOrDestroyed())
                Object.Destroy(shell.mesh);
        }
    }
    #endregion

    #region Groups
    Group Add(PQ parent)
    {
        var shell = AcquireShell(parent);
        var group = new Group { parent = parent, shell = shell };
        for (int i = 0; i < 4; ++i)
            group.children[i] = parent.subNodes[i];
        SyncRenderer(group);

        group.index = groupList.Count;
        groups.Add(parent, group);
        groupList.Add(group);
        Queue(group);
        return group;
    }

    // Moves a family between pending and ready as its children are built,
    // shown and hidden. False if it should be dissolved.
    bool UpdateReady(Group group)
    {
        if (!IsFamily(group.parent))
        {
            group.ready = false;
            SetMerged(group, false);
            return true;
        }

        if (group.ready)
            return StillRendersAlike(group);

        if (!DrawsAlike(group.parent))
            return false;

        // Builds can change materials, so match the renderer again.
        group.ready = true;
        SyncRenderer(group);
        Queue(group);
        return true;
    }

    void RemoveAt(int index)
    {
        var group = groupList[index];
        Dissolve(group);
        groups.Remove(group.parent);

        int last = groupList.Count - 1;
        var moved = groupList[last];
        groupList[index] = moved;
        moved.index = index;
        groupList.RemoveAt(last);
    }

    void Dissolve(Group group)
    {
        SetMerged(group, false);
        group.removed = true;
        ReleaseShell(group.shell);
        group.shell = null;
    }

    Shell AcquireShell(PQ parent)
    {
        Shell shell = null;
        while (pool.Count != 0 && shell is null)
        {
            var pooled = pool.Pop();
            if (pooled.gameObject.IsNotNullOrDestroyed() && pooled.mesh.IsNotNullOrDestroyed())
                shell = pooled;
        }

        if (shell is null)
        {
            var gameObject = new GameObject("QuadGroup");
            var mesh = new Mesh { name = "QuadGroup" };
            mesh.MarkDynamic();
            gameObject.AddComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = gameObject.AddComponent<MeshRenderer>();
            renderer.enabled = false;
            shell = new Shell
            {
                gameObject = gameObject,
                renderer = renderer,
                mesh = mesh,
            };
        }

        shell.gameObject.layer = parent.subNodes[0].gameObject.layer;
        shell.gameObject.transform.SetParent(parent.transform, false);
        return shell;
    }

    // Pooled shells wait under the sphere, so they outlive the quad they were
    // under and are destroyed with the sphere.
    void ReleaseShell(Shell shell)
    {
        if (shell.renderer.IsNotNullOrDestroyed())
            shell.renderer.enabled = false;

        if (pool.Count < MaxPooled && shell.gameObject.IsNotNullOrDestroyed())
        {
            shell.gameObject.transform.SetParent(batchPQS.transform, false);
            pool.Push(shell);
            return;
        }

        if (shell.gameObject.IsNotNullOrDestroyed())
            Object.Destroy(shell.gameObject);
        if (shell.mesh.IsNotNullOrDestroyed())
            Object.Destroy(shell.mesh);
    }

    static void SetMerged(Group group, bool merged)
    {
        if (group.merged == merged)
            return;
        group.merged = merged;

        var renderer = group.shell?.renderer;
        if (renderer.IsNotNullOrDestroyed())
            renderer.enabled = merged;

        foreach (var child in group.children)
        {
            if (child.IsNotNullOrDestroyed() && child.meshRenderer.IsNotNullOrDestroyed())
                child.meshRenderer.forceRenderingOff = merged;
        }
    }
    #endregion

    #region Checks
    // Whether this group's quads are still the leaf children of its parent.
    // Only reads PQ fields.
    static bool IsIntact(Group group)
    {
        if (!IsLeafFamily(group.parent))
            return false;

        var subNodes = group.parent.subNodes;
        for (int i = 0; i < 4; ++i)
        {
            if (!ReferenceEquals(subNodes[i], group.children[i]))
                return false;
        }
        return true;
    }

    // Whether all four children of this quad are leaves, built or not.
    static bool IsLeafFamily(PQ parent)
    {
        if (
            parent.IsNullOrDestroyed()
            || !parent.isSubdivided
            || !parent.isActive
            || parent.quadRoot is null
        )
            return false;

        var subNodes = parent.subNodes;
        if (subNodes is null || subNodes.Length != 4)
            return false;

        foreach (var child in subNodes)
        {
            if (child.IsNullOrDestroyed() || !child.isActive || child.isSubdivided)
                return false;
        }
        return true;
    }

    // Whether all four children of this quad are built, visible leaves.
    static bool IsFamily(PQ parent)
    {
        if (!IsLeafFamily(parent))
            return false;

        foreach (var child in parent.subNodes)
        {
            if (!child.isBuilt || !child.isVisible || child.isForcedInvisible)
                return false;
        }
        return true;
    }

    static bool HasUnbuiltChild(PQ parent)
    {
        foreach (var child in parent.subNodes)
        {
            if (!child.isBuilt)
                return true;
        }
        return false;
    }

    // Whether all four children render, and render the same way.
    static bool DrawsAlike(PQ parent)
    {
        if (!parent.gameObject.activeInHierarchy)
            return false;

        var first = parent.subNodes[0].meshRenderer;
        foreach (var child in parent.subNodes)
        {
            var renderer = child.meshRenderer;
            if (
                renderer.IsNullOrDestroyed()
                || !renderer.enabled
                || !child.gameObject.activeInHierarchy
                // Parallax swaps in its own mesh near the camera.
                || child.meshFilter.IsNullOrDestroyed()
                || !ReferenceEquals(child.meshFilter.sharedMesh, child.mesh)
            )
                return false;

            if (!ReferenceEquals(renderer, first) && !RendersSame(first, renderer))
                return false;
        }
        return true;
    }

    // The part of DrawsAlike that can change after a family forms: Parallax
    // swaps materials and meshes near the camera, EVE adds materials, and
    // Scatterer disables the stock ocean's renderers.
    static bool StillRendersAlike(Group group)
    {
        group.shell.renderer.GetSharedMaterials(MaterialsA);
        foreach (var child in group.children)
        {
            var renderer = child.meshRenderer;
            if (!renderer.enabled || !ReferenceEquals(child.meshFilter.sharedMesh, child.mesh))
                return false;

            renderer.GetSharedMaterials(MaterialsB);
            if (!SameList(MaterialsA, MaterialsB))
                return false;
        }
        return true;
    }

    static bool RendersSame(Renderer a, Renderer b) =>
        a.gameObject.layer == b.gameObject.layer
        && a.shadowCastingMode == b.shadowCastingMode
        && a.receiveShadows == b.receiveShadows
        && a.reflectionProbeUsage == b.reflectionProbeUsage
        && a.lightProbeUsage == b.lightProbeUsage
        && a.probeAnchor == b.probeAnchor
        && SameMaterials(a, b);

    static bool SameMaterials(Renderer a, Renderer b)
    {
        a.GetSharedMaterials(MaterialsA);
        b.GetSharedMaterials(MaterialsB);
        return SameList(MaterialsA, MaterialsB);
    }

    static bool SameList(List<Material> a, List<Material> b)
    {
        if (a.Count != b.Count)
            return false;

        for (int i = 0; i < a.Count; ++i)
        {
            if (!ReferenceEquals(a[i], b[i]))
                return false;
        }
        return true;
    }

    static void SyncRenderer(Group group)
    {
        var source = group.children[0].meshRenderer;
        var renderer = group.shell.renderer;

        if (!SameMaterials(source, renderer))
            renderer.sharedMaterials = source.sharedMaterials;
        if (renderer.shadowCastingMode != source.shadowCastingMode)
            renderer.shadowCastingMode = source.shadowCastingMode;
        if (renderer.receiveShadows != source.receiveShadows)
            renderer.receiveShadows = source.receiveShadows;
        if (renderer.reflectionProbeUsage != source.reflectionProbeUsage)
            renderer.reflectionProbeUsage = source.reflectionProbeUsage;
        if (renderer.lightProbeUsage != source.lightProbeUsage)
            renderer.lightProbeUsage = source.lightProbeUsage;
        if (renderer.probeAnchor != source.probeAnchor)
            renderer.probeAnchor = source.probeAnchor;
    }
    #endregion

    #region Copy
    // Every child's vertices are relative to its own (float) planet position
    // and quads don't rotate or scale relative to each other, so a child is
    // placed in the family mesh with a translation.
    static Vector3 SliceOffset(Group group, PQ child)
    {
        Vector3 position = child.positionPlanet;
        Vector3 origin = group.parent.positionPlanet;
        return position - origin;
    }

    // Reads each stale child back into its slice.
    static bool Copy(Group group)
    {
        var shell = group.shell;
        int count = group.children[0].mesh.GetVertexAttributes(AttributeScratch);
        if (shell.attributes is null || !SameLayout(shell.attributes, AttributeScratch, count))
        {
            // Creating the mesh for a new layout clears it.
            if (!CreateMesh(shell, AttributeScratch, count))
                return false;
            group.stale = 0xF;
        }

        for (int i = 0; i < 4; ++i)
        {
            if ((group.stale & (1 << i)) == 0)
                continue;

            if (!CopyChild(group, i))
                return false;

            group.stale &= ~(1 << i);
            CopiedQuads++;
        }

        UpdateBounds(group);
        return true;
    }

    static bool CopyChild(Group group, int index)
    {
        var shell = group.shell;
        var child = group.children[index];
        var source = child.mesh;
        int vertexCount = VertsPerQuad;
        if (source.vertexCount != vertexCount || source.subMeshCount != 1)
            return false;

        int count = source.GetVertexAttributes(AttributeScratch);
        if (!SameLayout(shell.attributes, AttributeScratch, count))
            return false;

        var offset = SliceOffset(group, child);
        var mesh = shell.mesh;
        int baseVertex = index * vertexCount;
        for (int stream = 0; stream < 4; ++stream)
        {
            switch (shell.kinds[stream])
            {
                case StreamKind.Single:
                    CopySingle(source, mesh, shell.single[stream], baseVertex, vertexCount);
                    break;

                case StreamKind.Interleaved:
                    CopyInterleaved(source, mesh, offset, stream, baseVertex, vertexCount);
                    break;

                case StreamKind.Floats:
                    CopyFloats(source, shell, offset, stream, baseVertex, vertexCount);
                    break;
            }
        }

        source.GetIndices(IndexScratch, 0);
        if (IndexScratch.Count != IndicesPerQuad)
            return false;
        WriteIndices(mesh, IndexScratch, index);

        var bounds = source.bounds;
        bounds.center += offset;
        group.bounds[index] = bounds;
        return true;
    }

    // Uploads a stream holding only this attribute straight from its getter.
    static void CopySingle(
        Mesh source,
        Mesh mesh,
        VertexAttributeDescriptor descriptor,
        int baseVertex,
        int vertexCount
    )
    {
        int stream = descriptor.stream;
        switch (descriptor.attribute)
        {
            case VertexAttribute.Normal:
                source.GetNormals(Vector3Scratch);
                mesh.SetVertexBufferData(
                    Vector3Scratch,
                    0,
                    baseVertex,
                    vertexCount,
                    stream,
                    UpdateFlags
                );
                return;

            case VertexAttribute.Tangent:
                source.GetTangents(Vector4Scratch);
                mesh.SetVertexBufferData(
                    Vector4Scratch,
                    0,
                    baseVertex,
                    vertexCount,
                    stream,
                    UpdateFlags
                );
                return;

            case VertexAttribute.Color:
                source.GetColors(ColorScratch);
                mesh.SetVertexBufferData(
                    ColorScratch,
                    0,
                    baseVertex,
                    vertexCount,
                    stream,
                    UpdateFlags
                );
                return;
        }

        int channel = descriptor.attribute - VertexAttribute.TexCoord0;
        switch (descriptor.dimension)
        {
            case 2:
                source.GetUVs(channel, Vector2Scratch);
                mesh.SetVertexBufferData(
                    Vector2Scratch,
                    0,
                    baseVertex,
                    vertexCount,
                    stream,
                    UpdateFlags
                );
                break;
            case 3:
                source.GetUVs(channel, Vector3Scratch);
                mesh.SetVertexBufferData(
                    Vector3Scratch,
                    0,
                    baseVertex,
                    vertexCount,
                    stream,
                    UpdateFlags
                );
                break;
            default:
                source.GetUVs(channel, Vector4Scratch);
                mesh.SetVertexBufferData(
                    Vector4Scratch,
                    0,
                    baseVertex,
                    vertexCount,
                    stream,
                    UpdateFlags
                );
                break;
        }
    }

    // Uploads BatchPQS's interleaved stream: position, color, uv0, uv2, uv3.
    static unsafe void CopyInterleaved(
        Mesh source,
        Mesh mesh,
        Vector3 offset,
        int stream,
        int baseVertex,
        int vertexCount
    )
    {
        source.GetVertices(Vector3Scratch);
        source.GetColors(ColorScratch);
        source.GetUVs(0, Vector2Scratch);
        source.GetUVs(2, Uv2Scratch);
        source.GetUVs(3, Uv3Scratch);

        var output = new NativeArray<InterleavedVertex>(
            vertexCount,
            Allocator.Temp,
            NativeArrayOptions.UninitializedMemory
        );
        InterleaveFunc ??= BurstUtil.MaybeCompileDelegate<InterleaveDelegate>(Interleave);
        fixed (Vector3* positions = NoAllocHelpers.ExtractArrayFromListT(Vector3Scratch))
        fixed (Color* colors = NoAllocHelpers.ExtractArrayFromListT(ColorScratch))
        fixed (Vector2* uv0s = NoAllocHelpers.ExtractArrayFromListT(Vector2Scratch))
        fixed (Vector2* uv2s = NoAllocHelpers.ExtractArrayFromListT(Uv2Scratch))
        fixed (Vector2* uv3s = NoAllocHelpers.ExtractArrayFromListT(Uv3Scratch))
        {
            InterleaveFunc(
                new(positions, vertexCount),
                new(colors, vertexCount),
                new(uv0s, vertexCount),
                new(uv2s, vertexCount),
                new(uv3s, vertexCount),
                new(output),
                offset
            );
        }

        mesh.SetVertexBufferData(output, 0, baseVertex, vertexCount, stream, UpdateFlags);
        output.Dispose();
    }

    delegate void InterleaveDelegate(
        in MemorySpan<Vector3> positions,
        in MemorySpan<Color> colors,
        in MemorySpan<Vector2> uv0s,
        in MemorySpan<Vector2> uv2s,
        in MemorySpan<Vector2> uv3s,
        in MemorySpan<InterleavedVertex> output,
        in Vector3 offset
    );

    static InterleaveDelegate InterleaveFunc;

    [BurstCompile(FloatMode = FloatMode.Fast)]
    static void Interleave(
        in MemorySpan<Vector3> positions,
        in MemorySpan<Color> colors,
        in MemorySpan<Vector2> uv0s,
        in MemorySpan<Vector2> uv2s,
        in MemorySpan<Vector2> uv3s,
        in MemorySpan<InterleavedVertex> output,
        in Vector3 offset
    )
    {
        for (int i = 0; i < output.Length; ++i)
        {
            output[i] = new InterleavedVertex
            {
                position = positions[i] + offset,
                color = colors[i],
                uv0 = uv0s[i],
                uv2 = uv2s[i],
                uv3 = uv3s[i],
            };
        }
    }

    // Interleaves any other stream as floats, one attribute at a time.
    static void CopyFloats(
        Mesh source,
        Shell shell,
        Vector3 offset,
        int stream,
        int baseVertex,
        int vertexCount
    )
    {
        int stride = shell.strides[stream];
        if (StreamScratch[stream].Length < stride * vertexCount)
            StreamScratch[stream] = new float[stride * vertexCount];

        foreach (var descriptor in shell.attributes)
        {
            if (descriptor.stream == stream)
                ReadAttribute(source, descriptor, shell, offset, vertexCount);
        }

        shell.mesh.SetVertexBufferData(
            StreamScratch[stream],
            0,
            baseVertex * stride,
            vertexCount * stride,
            stream,
            UpdateFlags
        );
    }

    // Writes one attribute of a quad into its stream's scratch buffer.
    static void ReadAttribute(
        Mesh source,
        VertexAttributeDescriptor descriptor,
        Shell shell,
        Vector3 offset,
        int vertexCount
    )
    {
        var attribute = descriptor.attribute;
        int dimension = descriptor.dimension;
        int stride = shell.strides[descriptor.stream];
        int at = shell.offsets[(int)attribute];
        var buffer = StreamScratch[descriptor.stream];

        switch (attribute)
        {
            case VertexAttribute.Position:
                source.GetVertices(Vector3Scratch);
                for (int i = 0; i < vertexCount; ++i, at += stride)
                {
                    var v = Vector3Scratch[i] + offset;
                    buffer[at] = v.x;
                    buffer[at + 1] = v.y;
                    buffer[at + 2] = v.z;
                }
                break;

            case VertexAttribute.Normal:
                source.GetNormals(Vector3Scratch);
                for (int i = 0; i < vertexCount; ++i, at += stride)
                {
                    var v = Vector3Scratch[i];
                    buffer[at] = v.x;
                    buffer[at + 1] = v.y;
                    buffer[at + 2] = v.z;
                }
                break;

            case VertexAttribute.Tangent:
                source.GetTangents(Vector4Scratch);
                WriteVector4s(Vector4Scratch, buffer, at, stride, 4, vertexCount);
                break;

            case VertexAttribute.Color:
                source.GetColors(ColorScratch);
                for (int i = 0; i < vertexCount; ++i, at += stride)
                {
                    var c = ColorScratch[i];
                    buffer[at] = c.r;
                    buffer[at + 1] = c.g;
                    buffer[at + 2] = c.b;
                    buffer[at + 3] = c.a;
                }
                break;

            default:
                int channel = attribute - VertexAttribute.TexCoord0;
                if (dimension == 2)
                {
                    source.GetUVs(channel, Vector2Scratch);
                    for (int i = 0; i < vertexCount; ++i, at += stride)
                    {
                        var v = Vector2Scratch[i];
                        buffer[at] = v.x;
                        buffer[at + 1] = v.y;
                    }
                }
                else
                {
                    source.GetUVs(channel, Vector4Scratch);
                    WriteVector4s(Vector4Scratch, buffer, at, stride, dimension, vertexCount);
                }
                break;
        }
    }

    static void WriteVector4s(
        List<Vector4> values,
        float[] buffer,
        int at,
        int stride,
        int dimension,
        int vertexCount
    )
    {
        for (int i = 0; i < vertexCount; ++i, at += stride)
        {
            var v = values[i];
            for (int c = 0; c < dimension; ++c)
                buffer[at + c] = v[c];
        }
    }

    static void WriteIndices(Mesh mesh, IList<int> indices, int index)
    {
        int indexCount = IndicesPerQuad;
        if (IndexSlice.Length != indexCount)
            IndexSlice = new ushort[indexCount];

        int baseVertex = index * VertsPerQuad;
        for (int i = 0; i < indexCount; ++i)
            IndexSlice[i] = (ushort)(indices[i] + baseVertex);
        mesh.SetIndexBufferData(IndexSlice, 0, index * indexCount, indexCount, UpdateFlags);
    }

    static void UpdateBounds(Group group)
    {
        var bounds = group.bounds[0];
        for (int i = 1; i < 4; ++i)
            bounds.Encapsulate(group.bounds[i]);

        var mesh = group.shell.mesh;
        mesh.SetSubMesh(
            0,
            new SubMeshDescriptor(0, 4 * IndicesPerQuad)
            {
                bounds = bounds,
                firstVertex = 0,
                vertexCount = 4 * VertsPerQuad,
            },
            UpdateFlags
        );
        mesh.bounds = bounds;
    }

    static bool SameLayout(VertexAttributeDescriptor[] a, VertexAttributeDescriptor[] b, int count)
    {
        if (a is null || a.Length != count)
            return false;

        for (int i = 0; i < count; ++i)
        {
            if (
                a[i].attribute != b[i].attribute
                || a[i].format != b[i].format
                || a[i].dimension != b[i].dimension
                || a[i].stream != b[i].stream
            )
                return false;
        }
        return true;
    }

    // Gives the family mesh its quads' vertex layout, four quads long. Every
    // attribute must be Float32 so quads can be copied as floats.
    static bool CreateMesh(Shell shell, VertexAttributeDescriptor[] attributes, int count)
    {
        System.Array.Clear(shell.strides, 0, shell.strides.Length);
        bool hasPosition = false;
        for (int i = 0; i < count; ++i)
        {
            var descriptor = attributes[i];
            if (descriptor.format != VertexAttributeFormat.Float32 || descriptor.stream >= 4)
                return false;

            hasPosition |= descriptor.attribute == VertexAttribute.Position;
            // Attributes are laid out in a stream in the order given.
            shell.offsets[(int)descriptor.attribute] = shell.strides[descriptor.stream];
            shell.strides[descriptor.stream] += descriptor.dimension;
        }
        if (!hasPosition)
            return false;

        shell.attributes = new VertexAttributeDescriptor[count];
        System.Array.Copy(attributes, shell.attributes, count);
        ClassifyStreams(shell);

        var mesh = shell.mesh;
        mesh.Clear();
        mesh.SetVertexBufferParams(4 * VertsPerQuad, shell.attributes);
        mesh.SetIndexBufferParams(4 * IndicesPerQuad, IndexFormat.UInt16);
        mesh.subMeshCount = 1;
        return true;
    }

    static void ClassifyStreams(Shell shell)
    {
        for (int stream = 0; stream < 4; ++stream)
        {
            int count = 0;
            foreach (var descriptor in shell.attributes)
            {
                if (descriptor.stream != stream)
                    continue;
                count++;
                shell.single[stream] = descriptor;
            }

            var single = shell.single[stream];
            if (count == 0)
                shell.kinds[stream] = StreamKind.None;
            else if (
                count == 1
                && single.attribute != VertexAttribute.Position
                && (single.attribute < VertexAttribute.TexCoord0 || single.dimension >= 2)
            )
                shell.kinds[stream] = StreamKind.Single;
            else if (IsInterleavedVertex(shell, stream))
                shell.kinds[stream] = StreamKind.Interleaved;
            else
                shell.kinds[stream] = StreamKind.Floats;
        }
    }

    // Whether the stream is laid out exactly as InterleavedVertex.
    static bool IsInterleavedVertex(Shell shell, int stream)
    {
        int matched = 0;
        foreach (var descriptor in shell.attributes)
        {
            if (descriptor.stream != stream)
                continue;

            int expected = descriptor.attribute switch
            {
                VertexAttribute.Position => 3,
                VertexAttribute.Color => 4,
                VertexAttribute.TexCoord0 => 2,
                VertexAttribute.TexCoord2 => 2,
                VertexAttribute.TexCoord3 => 2,
                _ => -1,
            };
            if (descriptor.dimension != expected)
                return false;
            matched++;
        }
        return matched == 5;
    }
    #endregion
}
