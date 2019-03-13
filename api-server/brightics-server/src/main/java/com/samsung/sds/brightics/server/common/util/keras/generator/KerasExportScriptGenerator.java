package com.samsung.sds.brightics.server.common.util.keras.generator;

import com.google.gson.JsonObject;
import com.samsung.sds.brightics.server.common.util.keras.KerasModelScriptGenerator;
import com.samsung.sds.brightics.server.common.util.keras.KerasScriptUtil;
import org.apache.commons.lang3.StringUtils;

public class KerasExportScriptGenerator extends KerasModelScriptGenerator {

    public KerasExportScriptGenerator(JsonObject model) throws Exception {
        super(model);
    }

    @Override
    protected void addDataLoad() throws Exception {
        script.add(KerasScriptUtil.makeTrainDataLoadScript(modelFlowData.getDataNodes(), true));
    }

    @Override
    protected void addGenerationModeSpecificScript() throws Exception {
        script.add(KerasScriptUtil.makeModelCompileScript(param));
        script.add(StringUtils.EMPTY);

        if (useFitGenerator()) {
            script.add(KerasScriptUtil.makeModelFitGeneratorScript(param, modelFlowData.getOutputNodes()));
        } else {
            script.add(KerasScriptUtil.makeModelFitScript(param, modelFlowData.getOutputNodes()));
        }
    }
}