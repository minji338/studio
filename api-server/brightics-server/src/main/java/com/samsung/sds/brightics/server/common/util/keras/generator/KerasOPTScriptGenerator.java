package com.samsung.sds.brightics.server.common.util.keras.generator;

import com.google.gson.JsonObject;
import com.samsung.sds.brightics.server.common.util.keras.KerasScriptUtil;
import org.apache.commons.lang3.StringUtils;

public class KerasOPTScriptGenerator extends KerasTrainScriptGenerator {

    public static final String SINGLE_INDENT = "    ";

    public KerasOPTScriptGenerator(JsonObject model, String jid) throws Exception {
        super(model, jid);
    }

    @Override
    public String getFunctionalScript() throws Exception {
        clearScript();

        addFunctionalImport();
        addLayersImport();
        addHyperOptImport();

        addDataLoad();

        addHyperOptModel();

        addGenerationModeSpecificScript();

        return script.toString();
    }

    private void addHyperOptImport() {
        script.add("from hyperopt import Trials, STATUS_OK, tpe");
        script.add("from hyperas import optim");
        script.add("from hyperas.distributions import choice, uniform");
        script.add(StringUtils.EMPTY);
    }

    @Override
    protected void addDataLoad() throws Exception {
        super.addDataLoad();
        script.add(StringUtils.EMPTY);

        script.add("def data():");
        script.add(String.format("%sreturn X_train, Y_train, X_train, Y_train", SINGLE_INDENT));
        script.add(StringUtils.EMPTY);
    }

    private void addHyperOptModel() throws Exception {
        script.add("def create_model(X_train, Y_train, X_test, Y_test):");
        script.add(KerasScriptUtil.makeFunctionalModelScript(modelFlowData, SINGLE_INDENT)).add(StringUtils.EMPTY);
        script.add(KerasScriptUtil.makeModelCompileScript(param, SINGLE_INDENT));
        script.add(StringUtils.EMPTY);
        script.add(KerasScriptUtil.makeModelFitScript(param, modelFlowData.getOutputNodes(), SINGLE_INDENT));
        script.add(String.format("%sacc = np.amax(result.history['acc'])", SINGLE_INDENT));
        script.add(String.format("%sreturn {'loss': -acc, 'status': STATUS_OK, 'model': model}", SINGLE_INDENT));
        script.add(StringUtils.EMPTY);
    }

    @Override
    protected void addGenerationModeSpecificScript() throws Exception {
        script.add("best_run, best_model = optim.minimize(mode=create_model, data=data, algo=tpe.suggest, max_evals=3, trials=Trials())");
    }
}
