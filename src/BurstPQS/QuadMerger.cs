using System.Collections.Generic;
using BurstPQS.Util;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Rendering;

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

    static readonly ProfilerMarker ReconcileMarker = new("QuadMerger.Reconcile");
    static readonly ProfilerMarker RebuildMarker = new("QuadMerger.Rebuild");

    sealed class Group
    {
        public PQ parent;
        public readonly PQ[] children = new PQ[4];
        public GameObject gameObject;
        public MeshRenderer renderer;
        public Mesh mesh;

        // Set when a child's mesh changes. Cleared by a rebuild.
        public bool dirty = true;

        // False if the children could not be merged; they draw themselves
        // until one of them changes again.
        public bool merged;
    }

    readonly Dictionary<PQ, Group> groups = [];
    readonly List<PQ> removeScratch = [];

    static readonly List<Material> MaterialsA = [];
    static readonly List<Material> MaterialsB = [];

    static readonly List<Vector3> Positions = [];
    static readonly List<Vector3> Normals = [];
    static readonly List<Vector4> Tangents = [];
    static readonly List<Color> Colors = [];
    static readonly List<Vector2>[] Uv2s =
    [
        [],
        [],
        [],
        [],
    ];
    static readonly List<Vector4>[] Uv4s =
    [
        [],
        [],
        [],
        [],
    ];
    static readonly List<int> Indices = [];

    static readonly List<Vector3> Vector3Scratch = [];
    static readonly List<Vector4> Vector4Scratch = [];
    static readonly List<Color> ColorScratch = [];
    static readonly List<Vector2> Vector2Scratch = [];
    static readonly List<int> IndexScratch = [];
    static readonly int[] UvDimensions = new int[4];

    // A child of this quad's mesh changed, so its family must be rebuilt.
    internal void MarkDirty(PQ quad)
    {
        var parent = quad.parent;
        if (parent is not null && groups.TryGetValue(parent, out var group))
            group.dirty = true;
    }

    // Runs once a frame, after the quad tree has been updated.
    internal void Reconcile(List<PQ> activeQuads)
    {
        bool canMerge = Enabled && batchPQS.CanMergeQuads;
        if (!canMerge && groups.Count == 0)
            return;

        using var scope = ReconcileMarker.Auto();

        foreach (var pair in groups)
        {
            if (!canMerge || !IsSameFamily(pair.Key, pair.Value) || !IsComplete(pair.Key))
                removeScratch.Add(pair.Key);
        }

        foreach (var parent in removeScratch)
        {
            Dissolve(groups[parent]);
            groups.Remove(parent);
        }
        removeScratch.Clear();

        if (!canMerge)
            return;

        foreach (var quad in activeQuads)
        {
            if (
                quad.IsNotNullOrDestroyed()
                && quad.isSubdivided
                && !groups.ContainsKey(quad)
                && IsComplete(quad)
            )
                groups.Add(quad, Create(quad));
        }

        foreach (var group in groups.Values)
        {
            if (group.dirty)
            {
                group.dirty = false;
                group.merged = Rebuild(group);
            }

            if (group.merged)
                SyncRenderer(group);
            SetMerged(group, group.merged);
        }
    }

    internal void Clear()
    {
        foreach (var group in groups.Values)
            Dissolve(group);
        groups.Clear();
    }

    static bool IsSameFamily(PQ parent, Group group)
    {
        if (parent.IsNullOrDestroyed())
            return false;

        var subNodes = parent.subNodes;
        for (int i = 0; i < 4; ++i)
        {
            if (!ReferenceEquals(subNodes[i], group.children[i]))
                return false;
        }
        return true;
    }

    // Whether all four children of this quad draw, and draw the same way.
    static bool IsComplete(PQ parent)
    {
        if (
            parent.IsNullOrDestroyed()
            || !parent.isSubdivided
            || !parent.isActive
            || parent.quadRoot is null
            || !parent.gameObject.activeInHierarchy
        )
            return false;

        var subNodes = parent.subNodes;
        if (subNodes is null || subNodes.Length != 4)
            return false;

        var first = subNodes[0];
        for (int i = 0; i < 4; ++i)
        {
            if (!IsDrawnLeaf(subNodes[i]))
                return false;
            if (i != 0 && !DrawsSame(first.meshRenderer, subNodes[i].meshRenderer))
                return false;
        }
        return true;
    }

    static bool IsDrawnLeaf(PQ quad) =>
        quad.IsNotNullOrDestroyed()
        && quad.isActive
        && quad.isBuilt
        && !quad.isSubdivided
        && quad.isVisible
        && !quad.isForcedInvisible
        && quad.meshRenderer.IsNotNullOrDestroyed()
        && quad.meshRenderer.enabled
        && quad.gameObject.activeInHierarchy
        // Parallax swaps in its own mesh near the camera.
        && quad.meshFilter.IsNotNullOrDestroyed()
        && ReferenceEquals(quad.meshFilter.sharedMesh, quad.mesh);

    static bool DrawsSame(Renderer a, Renderer b) =>
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
        if (MaterialsA.Count != MaterialsB.Count)
            return false;

        for (int i = 0; i < MaterialsA.Count; ++i)
        {
            if (!ReferenceEquals(MaterialsA[i], MaterialsB[i]))
                return false;
        }
        return true;
    }

    static Group Create(PQ parent)
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
        return group;
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
        if (group.renderer.IsNotNullOrDestroyed() && group.renderer.enabled != merged)
            group.renderer.enabled = merged;

        foreach (var child in group.children)
        {
            if (child.IsNullOrDestroyed() || child.meshRenderer.IsNullOrDestroyed())
                continue;
            if (child.meshRenderer.forceRenderingOff != merged)
                child.meshRenderer.forceRenderingOff = merged;
        }
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

    // Copies the four child meshes into the group mesh. Every child's vertices
    // are relative to its own (float) planet position and quads don't rotate
    // or scale relative to each other, so merging is a translation per child.
    static bool Rebuild(Group group)
    {
        using var scope = RebuildMarker.Auto();

        var first = group.children[0].mesh;
        int vertexCount = first.vertexCount;
        bool hasNormals = first.HasVertexAttribute(VertexAttribute.Normal);
        bool hasTangents = first.HasVertexAttribute(VertexAttribute.Tangent);
        bool hasColors = first.HasVertexAttribute(VertexAttribute.Color);
        var uvDimensions = UvDimensions;
        for (int channel = 0; channel < 4; ++channel)
            uvDimensions[channel] = UvDimension(first, channel);

        Positions.Clear();
        Normals.Clear();
        Tangents.Clear();
        Colors.Clear();
        Indices.Clear();
        for (int channel = 0; channel < 4; ++channel)
        {
            Uv2s[channel].Clear();
            Uv4s[channel].Clear();
        }

        Vector3 origin = group.parent.positionPlanet;
        Bounds bounds = default;

        for (int i = 0; i < 4; ++i)
        {
            var child = group.children[i];
            var mesh = child.mesh;
            if (
                mesh.vertexCount != vertexCount
                || mesh.subMeshCount != 1
                || mesh.HasVertexAttribute(VertexAttribute.Normal) != hasNormals
                || mesh.HasVertexAttribute(VertexAttribute.Tangent) != hasTangents
                || mesh.HasVertexAttribute(VertexAttribute.Color) != hasColors
            )
                return false;
            for (int channel = 0; channel < 4; ++channel)
            {
                if (UvDimension(mesh, channel) != uvDimensions[channel])
                    return false;
            }

            Vector3 position = child.positionPlanet;
            var offset = position - origin;
            int baseVertex = Positions.Count;

            mesh.GetVertices(Vector3Scratch);
            foreach (var vertex in Vector3Scratch)
                Positions.Add(vertex + offset);

            if (hasNormals)
            {
                mesh.GetNormals(Vector3Scratch);
                Normals.AddRange(Vector3Scratch);
            }
            if (hasTangents)
            {
                mesh.GetTangents(Vector4Scratch);
                Tangents.AddRange(Vector4Scratch);
            }
            if (hasColors)
            {
                mesh.GetColors(ColorScratch);
                Colors.AddRange(ColorScratch);
            }
            for (int channel = 0; channel < 4; ++channel)
            {
                switch (uvDimensions[channel])
                {
                    case 0:
                        break;
                    case 2:
                        mesh.GetUVs(channel, Vector2Scratch);
                        Uv2s[channel].AddRange(Vector2Scratch);
                        break;
                    default:
                        mesh.GetUVs(channel, Vector4Scratch);
                        Uv4s[channel].AddRange(Vector4Scratch);
                        break;
                }
            }

            mesh.GetIndices(IndexScratch, 0);
            foreach (var index in IndexScratch)
                Indices.Add(index + baseVertex);

            var childBounds = mesh.bounds;
            childBounds.center += offset;
            if (i == 0)
                bounds = childBounds;
            else
                bounds.Encapsulate(childBounds);
        }

        var merged = group.mesh;
        merged.Clear();
        merged.SetVertices(Positions);
        if (hasNormals)
            merged.SetNormals(Normals);
        if (hasTangents)
            merged.SetTangents(Tangents);
        if (hasColors)
            merged.SetColors(Colors);
        for (int channel = 0; channel < 4; ++channel)
        {
            switch (uvDimensions[channel])
            {
                case 0:
                    break;
                case 2:
                    merged.SetUVs(channel, Uv2s[channel]);
                    break;
                default:
                    merged.SetUVs(channel, Uv4s[channel]);
                    break;
            }
        }
        merged.SetIndices(Indices, MeshTopology.Triangles, 0, false);
        merged.bounds = bounds;
        return true;
    }

    static int UvDimension(Mesh mesh, int channel)
    {
        var attribute = VertexAttribute.TexCoord0 + channel;
        return mesh.HasVertexAttribute(attribute) ? mesh.GetVertexAttributeDimension(attribute) : 0;
    }
}
