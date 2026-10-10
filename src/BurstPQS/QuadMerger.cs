using System.Collections.Generic;
using BurstPQS.Util;
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
internal sealed class QuadMerger(BatchPQS batchPQS)
{
    internal static bool Enabled = true;

    // Renderer checks cross into native code, so only this many families are
    // rechecked each frame. PQ fields are checked for every family every frame.
    const int ChecksPerFrame = 16;

    // A family that failed the renderer checks is retried after this many frames.
    const int RejectFrames = 60;

    // Main thread time per frame for copying child meshes. Families still
    // waiting draw their children until they are copied.
    static readonly long CopyBudget = Stopwatch.Frequency / 4000;

    // Accumulated cost, for profiling.
    internal static long IntactTicks;
    internal static long RecheckTicks;
    internal static long ScanTicks;
    internal static long CopyTicks;
    internal static int CopiedQuads;
    internal static int Frames;
    static int lastFrame = -1;

    static readonly ProfilerMarker ReconcileMarker = new("QuadMerger.Reconcile");

    sealed class Group
    {
        public PQ parent;
        public readonly PQ[] children = new PQ[4];
        public readonly Bounds[] bounds = new Bounds[4];
        public GameObject gameObject;
        public MeshRenderer renderer;
        public Mesh mesh;

        // The vertex layout the mesh was created with, or -1.
        public int layout = -1;

        // Children whose part of the mesh is out of date.
        public int stale = 0xF;

        // Whether the merged mesh is drawn instead of the children.
        public bool merged;

        // Set when a child could not be copied, until a child changes again.
        public bool failed;
    }

    readonly Dictionary<PQ, Group> groups = [];
    readonly List<Group> groupList = [];
    readonly HashSet<PQ> rejected = [];
    int frame;
    int checkCursor;
    int copyCursor;

    static readonly List<Material> MaterialsA = [];
    static readonly List<Material> MaterialsB = [];
    static readonly List<VertexAttributeDescriptor> Attributes = [];
    static readonly List<Vector2> Vector2Scratch = [];
    static readonly List<Vector3> Vector3Scratch = [];
    static readonly List<Vector4> Vector4Scratch = [];
    static readonly List<Color> ColorScratch = [];
    static readonly List<int> IndexScratch = [];
    static float[] InterleavedScratch = [];
    static ushort[] IndexSlice = [];

    const MeshUpdateFlags UpdateFlags =
        MeshUpdateFlags.DontValidateIndices
        | MeshUpdateFlags.DontRecalculateBounds
        | MeshUpdateFlags.DontResetBoneBounds;

    static int VertsPerQuad => PQS.cacheVertCount;
    static int IndicesPerQuad => PQS.cacheIndices[0].Length;

    // A quad's mesh changed, so its part of the family mesh must be copied again.
    internal void MarkDirty(PQ quad)
    {
        var parent = quad.parent;
        if (parent is null || !groups.TryGetValue(parent, out var group))
            return;

        for (int i = 0; i < 4; ++i)
        {
            if (ReferenceEquals(group.children[i], quad))
            {
                group.stale |= 1 << i;
                group.failed = false;
            }
        }
    }

    // Runs once a frame, after the quad tree has been updated.
    internal void Reconcile(List<PQ> activeQuads)
    {
        if (!Enabled || !batchPQS.CanMergeQuads)
        {
            if (groupList.Count != 0)
                Clear();
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
            rejected.Clear();

        for (int i = groupList.Count - 1; i >= 0; --i)
        {
            if (!IsIntact(groupList[i]))
                RemoveAt(i);
        }

        long recheckStart = Stopwatch.GetTimestamp();
        IntactTicks += recheckStart - start;

        // A family that no longer matches is dissolved, and formed again below
        // if its children still render alike.
        int checks = Mathf.Min(ChecksPerFrame, groupList.Count);
        for (int n = 0; n < checks && groupList.Count != 0; ++n)
        {
            checkCursor %= groupList.Count;
            if (StillRendersAlike(groupList[checkCursor]))
                checkCursor++;
            else
                RemoveAt(checkCursor);
        }

        long scanStart = Stopwatch.GetTimestamp();
        RecheckTicks += scanStart - recheckStart;

        foreach (var quad in activeQuads)
        {
            if (
                quad.IsNullOrDestroyed()
                || !quad.isSubdivided
                || groups.ContainsKey(quad)
                || !IsFamily(quad)
                || rejected.Contains(quad)
            )
                continue;

            if (DrawsAlike(quad))
                Add(quad);
            else
                rejected.Add(quad);
        }

        long copyStart = Stopwatch.GetTimestamp();
        ScanTicks += copyStart - scanStart;

        long deadline = copyStart + CopyBudget;
        int count = groupList.Count;
        for (int n = 0; n < count; ++n)
        {
            var group = groupList[(copyCursor + n) % count];
            if (group.stale == 0 || group.failed)
                continue;

            if (Stopwatch.GetTimestamp() > deadline)
            {
                SetMerged(group, false);
                continue;
            }

            if (!Copy(group))
            {
                group.failed = true;
                SetMerged(group, false);
                continue;
            }

            SetMerged(group, true);
        }
        if (count != 0)
            copyCursor = (copyCursor + 1) % count;

        CopyTicks += Stopwatch.GetTimestamp() - copyStart;
    }

    internal void Clear()
    {
        foreach (var group in groupList)
            Dissolve(group);
        groupList.Clear();
        groups.Clear();
    }

    void Add(PQ parent)
    {
        var gameObject = new GameObject("QuadGroup");
        gameObject.layer = parent.subNodes[0].gameObject.layer;
        gameObject.transform.SetParent(parent.transform, false);

        var mesh = new Mesh { name = "QuadGroup" };
        mesh.MarkDynamic();
        gameObject.AddComponent<MeshFilter>().sharedMesh = mesh;

        var group = new Group
        {
            parent = parent,
            gameObject = gameObject,
            renderer = gameObject.AddComponent<MeshRenderer>(),
            mesh = mesh,
        };
        group.renderer.enabled = false;
        for (int i = 0; i < 4; ++i)
            group.children[i] = parent.subNodes[i];
        SyncRenderer(group);

        groups.Add(parent, group);
        groupList.Add(group);
    }

    void RemoveAt(int index)
    {
        var group = groupList[index];
        Dissolve(group);
        groups.Remove(group.parent);

        int last = groupList.Count - 1;
        groupList[index] = groupList[last];
        groupList.RemoveAt(last);
    }

    static void Dissolve(Group group)
    {
        SetMerged(group, false);
        if (group.gameObject.IsNotNullOrDestroyed())
            Object.Destroy(group.gameObject);
        if (group.mesh.IsNotNullOrDestroyed())
            Object.Destroy(group.mesh);
    }

    static void SetMerged(Group group, bool merged)
    {
        if (group.merged == merged)
            return;
        group.merged = merged;

        if (group.renderer.IsNotNullOrDestroyed())
            group.renderer.enabled = merged;

        foreach (var child in group.children)
        {
            if (child.IsNotNullOrDestroyed() && child.meshRenderer.IsNotNullOrDestroyed())
                child.meshRenderer.forceRenderingOff = merged;
        }
    }

    // Whether this group's quads are still the visible children of its parent.
    // Only reads PQ fields, so it is cheap enough to run on every group.
    static bool IsIntact(Group group)
    {
        if (!IsFamily(group.parent))
            return false;

        var subNodes = group.parent.subNodes;
        for (int i = 0; i < 4; ++i)
        {
            if (!ReferenceEquals(subNodes[i], group.children[i]))
                return false;
        }
        return true;
    }

    // Whether all four children of this quad are visible leaves.
    static bool IsFamily(PQ parent)
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
            if (
                child.IsNullOrDestroyed()
                || !child.isActive
                || !child.isBuilt
                || child.isSubdivided
                || !child.isVisible
                || child.isForcedInvisible
            )
                return false;
        }
        return true;
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
        group.renderer.GetSharedMaterials(MaterialsA);
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
        var renderer = group.renderer;

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

        int layer = source.gameObject.layer;
        if (group.gameObject.layer != layer)
            group.gameObject.layer = layer;
    }

    // Copies each stale child into its part of the family mesh. Every child's
    // vertices are relative to its own (float) planet position and quads don't
    // rotate or scale relative to each other, so a child is copied with a
    // translation.
    static bool Copy(Group group)
    {
        // Creating the mesh for a new layout clears it, so every child is copied again.
        int layout = Layout(group.children[0].mesh);
        if (layout < 0)
            return false;
        if (layout != group.layout)
        {
            CreateMesh(group.mesh, layout);
            group.layout = layout;
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

        var bounds = group.bounds[0];
        for (int i = 1; i < 4; ++i)
            bounds.Encapsulate(group.bounds[i]);

        group.mesh.SetSubMesh(
            0,
            new SubMeshDescriptor(0, 4 * IndicesPerQuad)
            {
                bounds = bounds,
                firstVertex = 0,
                vertexCount = 4 * VertsPerQuad,
            },
            UpdateFlags
        );
        group.mesh.bounds = bounds;
        return true;
    }

    static bool CopyChild(Group group, int index)
    {
        var child = group.children[index];
        var source = child.mesh;
        int vertexCount = VertsPerQuad;
        if (source.vertexCount != vertexCount || source.subMeshCount != 1)
            return false;

        int layout = group.layout;
        if (Layout(source) != layout)
            return false;

        var mesh = group.mesh;
        int baseVertex = index * vertexCount;
        Vector3 position = child.positionPlanet;
        Vector3 origin = group.parent.positionPlanet;
        var offset = position - origin;

        source.GetVertices(Vector3Scratch);
        for (int i = 0; i < vertexCount; ++i)
            Vector3Scratch[i] += offset;
        mesh.SetVertexBufferData(Vector3Scratch, 0, baseVertex, vertexCount, 0, UpdateFlags);

        source.GetNormals(Vector3Scratch);
        mesh.SetVertexBufferData(Vector3Scratch, 0, baseVertex, vertexCount, 1, UpdateFlags);

        int stream = 2;
        if (HasTangents(layout))
        {
            source.GetTangents(Vector4Scratch);
            mesh.SetVertexBufferData(Vector4Scratch, 0, baseVertex, vertexCount, 2, UpdateFlags);
            stream = 3;
        }

        int stride = InterleavedStride(layout);
        if (stride != 0)
        {
            CopyInterleaved(source, layout, stride, vertexCount);
            mesh.SetVertexBufferData(
                InterleavedScratch,
                0,
                baseVertex * stride,
                vertexCount * stride,
                stream,
                UpdateFlags
            );
        }

        int indexCount = IndicesPerQuad;
        source.GetIndices(IndexScratch, 0);
        if (IndexScratch.Count > indexCount)
            return false;
        if (IndexSlice.Length != indexCount)
            IndexSlice = new ushort[indexCount];
        for (int i = 0; i < IndexScratch.Count; ++i)
            IndexSlice[i] = (ushort)(IndexScratch[i] + baseVertex);
        // Pad with degenerate triangles.
        for (int i = IndexScratch.Count; i < indexCount; ++i)
            IndexSlice[i] = (ushort)baseVertex;
        mesh.SetIndexBufferData(IndexSlice, 0, index * indexCount, indexCount, UpdateFlags);

        var bounds = source.bounds;
        bounds.center += offset;
        group.bounds[index] = bounds;
        return true;
    }

    // Packs whether the mesh has tangents and colors, and each UV channel's
    // dimension. Positions and normals are required.
    static int Layout(Mesh mesh)
    {
        if (
            !mesh.HasVertexAttribute(VertexAttribute.Position)
            || !mesh.HasVertexAttribute(VertexAttribute.Normal)
        )
            return -1;

        int layout = 0;
        if (mesh.HasVertexAttribute(VertexAttribute.Tangent))
            layout |= 1;
        if (mesh.HasVertexAttribute(VertexAttribute.Color))
            layout |= 2;
        for (int channel = 0; channel < 4; ++channel)
            layout |= UvDimension(mesh, channel) << (2 + 3 * channel);
        return layout;
    }

    static bool HasTangents(int layout) => (layout & 1) != 0;

    static bool HasColors(int layout) => (layout & 2) != 0;

    static int UvDimension(int layout, int channel) => (layout >> (2 + 3 * channel)) & 7;

    static int UvDimension(Mesh mesh, int channel)
    {
        var attribute = VertexAttribute.TexCoord0 + channel;
        return mesh.HasVertexAttribute(attribute) ? mesh.GetVertexAttributeDimension(attribute) : 0;
    }

    // Floats per vertex in the stream holding colors and UVs.
    static int InterleavedStride(int layout)
    {
        int stride = HasColors(layout) ? 4 : 0;
        for (int channel = 0; channel < 4; ++channel)
            stride += UvDimension(layout, channel);
        return stride;
    }

    // Positions, normals and tangents each get their own stream so a child can
    // be written in place. Colors and UVs share the last one.
    static void CreateMesh(Mesh mesh, int layout)
    {
        Attributes.Clear();
        Attributes.Add(new(VertexAttribute.Position, VertexAttributeFormat.Float32, 3, 0));
        Attributes.Add(new(VertexAttribute.Normal, VertexAttributeFormat.Float32, 3, 1));

        int stream = 2;
        if (HasTangents(layout))
        {
            Attributes.Add(new(VertexAttribute.Tangent, VertexAttributeFormat.Float32, 4, 2));
            stream = 3;
        }
        if (HasColors(layout))
            Attributes.Add(new(VertexAttribute.Color, VertexAttributeFormat.Float32, 4, stream));
        for (int channel = 0; channel < 4; ++channel)
        {
            int dimension = UvDimension(layout, channel);
            if (dimension != 0)
                Attributes.Add(
                    new(
                        VertexAttribute.TexCoord0 + channel,
                        VertexAttributeFormat.Float32,
                        dimension,
                        stream
                    )
                );
        }

        mesh.Clear();
        mesh.SetVertexBufferParams(4 * VertsPerQuad, Attributes.ToArray());
        mesh.SetIndexBufferParams(4 * IndicesPerQuad, IndexFormat.UInt16);
        mesh.subMeshCount = 1;
    }

    static void CopyInterleaved(Mesh source, int layout, int stride, int vertexCount)
    {
        if (InterleavedScratch.Length < stride * vertexCount)
            InterleavedScratch = new float[stride * vertexCount];

        int offset = 0;
        if (HasColors(layout))
        {
            source.GetColors(ColorScratch);
            for (int i = 0; i < vertexCount; ++i)
            {
                var color = ColorScratch[i];
                int at = i * stride;
                InterleavedScratch[at] = color.r;
                InterleavedScratch[at + 1] = color.g;
                InterleavedScratch[at + 2] = color.b;
                InterleavedScratch[at + 3] = color.a;
            }
            offset = 4;
        }

        for (int channel = 0; channel < 4; ++channel)
        {
            int dimension = UvDimension(layout, channel);
            if (dimension == 0)
                continue;

            if (dimension == 2)
            {
                source.GetUVs(channel, Vector2Scratch);
                for (int i = 0; i < vertexCount; ++i)
                {
                    var uv = Vector2Scratch[i];
                    int at = i * stride + offset;
                    InterleavedScratch[at] = uv.x;
                    InterleavedScratch[at + 1] = uv.y;
                }
            }
            else
            {
                source.GetUVs(channel, Vector4Scratch);
                for (int i = 0; i < vertexCount; ++i)
                {
                    var uv = Vector4Scratch[i];
                    int at = i * stride + offset;
                    for (int c = 0; c < dimension; ++c)
                        InterleavedScratch[at + c] = uv[c];
                }
            }
            offset += dimension;
        }
    }
}
