//@ts-ignore
import wasmSimd from "./build/raw-simd/common_lingua_wasm_bg.wasm";
//@ts-ignore
import wasm from "./build/raw/common_lingua_wasm_bg.wasm";
import { simd } from "wasm-feature-detect";
//@ts-ignore
import initSimd, { detect as wasmSimdDetect, load as wasmSimdLoad } from "./build/raw-simd/common_lingua_wasm.js";
//@ts-ignore
import init, { detect as wasmDetect, load as wasmLoad } from "./build/raw/common_lingua_wasm.js";
//@ts-ignore
import st from "../../common-lingua/src/weight.safetensors";

const canSimd = await simd();
await (canSimd ? initSimd : init)({ module_or_path: canSimd ? wasmSimd : wasm });
(canSimd ? wasmSimdLoad : wasmLoad)(st);

export function detect(s: string): Record<string, number> {
  return Object.fromEntries((canSimd ? wasmSimdDetect : wasmDetect)(s));
}
