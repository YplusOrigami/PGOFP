import {finalizePresentation} from '/Users/wuaijia/.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations/container_tools/artifact_tool_utils.mjs';
const base='/private/tmp/tokamacon-structure-final';
const skill='/Users/wuaijia/.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations';
const tableOwners=[5,10,12,13,17];
const result=await finalizePresentation({
 workspaceDir:base,candidatePath:base+'/build/candidate.pptx',
 finalPath:base+'/output/TokamaCon_结构增强版_18页_30分钟.pptx',
 pythonExecutable:'/Users/wuaijia/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:18,requiredNativeTableOwnerSlides:tableOwners,
 requiredNativeChartOwnerSlides:[8,10,14,15,16],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:['PingFang SC']},verifyArtifactToolImport:true,
 receiptPath:base+'/build/validation.json'
});
console.log(JSON.stringify({finalPath:result.finalPath,sha256:result.finalSha256,slideCount:result.packageIntegrity.slide_count,integrity:result.packageIntegrity.status,layoutFindings:result.presentationLayout.findingCount,chartValidation:result.nativeChartValidation.passed}));
