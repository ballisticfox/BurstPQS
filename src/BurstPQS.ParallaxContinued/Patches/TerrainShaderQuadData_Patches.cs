using System.Collections.Generic;
using System.Reflection;
using HarmonyLib;
using Parallax;

namespace BurstPQS.ParallaxContinued.Patches;

// Parallax swaps a quad's material and probe anchor as the camera moves, so
// merged draws that include the quad need rebuilding.
[HarmonyPatch]
internal static class TerrainShaderQuadData_RendererChange_Patch
{
    static IEnumerable<MethodInfo> TargetMethods()
    {
        yield return AccessTools.Method(
            typeof(TerrainShaderQuadData),
            nameof(TerrainShaderQuadData.SwapMaterial)
        );
        yield return AccessTools.Method(
            typeof(TerrainShaderQuadData),
            nameof(TerrainShaderQuadData.OutOfRange)
        );
        yield return AccessTools.Method(
            typeof(TerrainShaderQuadData),
            nameof(TerrainShaderQuadData.SetReflectionProbeAnchor)
        );
    }

    static void Postfix(TerrainShaderQuadData __instance) =>
        BatchPQS.OnQuadRendererChanged(__instance.quad);
}
