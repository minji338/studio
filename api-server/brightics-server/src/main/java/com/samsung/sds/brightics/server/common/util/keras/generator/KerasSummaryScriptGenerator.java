package com.samsung.sds.brightics.server.common.util.keras.generator;

import com.google.gson.JsonObject;
import com.samsung.sds.brightics.server.common.util.keras.KerasModelScriptGenerator;
import com.samsung.sds.brightics.server.common.util.keras.KerasScriptUtil;

public class KerasSummaryScriptGenerator extends KerasModelScriptGenerator {

    public KerasSummaryScriptGenerator(JsonObject model, String jid) throws Exception {
        super(model, jid);
    }

    @Override
    protected void addDataLoad() {
        // DO NOTHING
    }

    @Override
    protected void addGenerationModeSpecificScript() {
        script.add(KerasScriptUtil.makeModelSummaryWriteScript(jid));
    }
}