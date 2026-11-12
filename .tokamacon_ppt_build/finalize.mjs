import {finalizePresentation} from '/Users/wuaijia/.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations/container_tools/artifact_tool_utils.mjs';
const base='/private/tmp/tokamacon-finalize';
const skill='/Users/wuaijia/.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations';
const result=await finalizePresentation({
 workspaceDir:base,
 candidatePath:base+'/.tokamacon_ppt_build/candidate.pptx',
 finalPath:base+'/output/ppt/TokamaCon_15页_30分钟.pptx',
 pythonExecutable:'/Users/wuaijia/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[3,8,9,10,14].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:15,
 requiredNativeTableOwnerSlides:[3,8,9,10,14],
 requiredNativeChartOwnerSlides:[6,8,11,12,13],
 materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:['PingFang SC']},
 verifyArtifactToolImport:true,
 receiptPath:base+'/.tokamacon_ppt_build/validation.json'
});
console.log(JSON.stringify(result));
