package com.samsung.sds.brightics.server.common.util.keras.flow;

import com.google.gson.JsonObject;
import com.samsung.sds.brightics.server.common.util.keras.model.KerasLayers;
import org.apache.commons.lang3.StringUtils;

import java.util.Arrays;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * For Layers have inner layer as listed below
 *   * RNN
 *   * TimeDistributed
 *   * Bidirectional
 */
public class KerasFlowWrapperNode extends KerasFlowLayerNode {

    private JsonObject innerObject;
    private String innerOperation;
    private JsonObject innerParam;

    private KerasLayers innerLayer = null;

    public KerasFlowWrapperNode(String fid, JsonObject function, String innerObjectKey) {
        super(fid, function);

        this.innerObject = param.getAsJsonObject(innerObjectKey);
        this.innerOperation = innerObject.get("name").getAsString();
        this.innerParam = innerObject.getAsJsonObject("param");

        if ("Custom".equalsIgnoreCase(innerOperation)) {
            param.addProperty(innerObjectKey, innerParam.get("function").getAsString());
        } else {
            innerLayer = KerasLayers.of(innerOperation);
            param.addProperty(innerObjectKey, innerLayer.getLayerScript(innerParam));
        }
    }

    @Override
    public boolean hasScript() {
        return innerParam.get("script") != null && StringUtils.isNotBlank(innerParam.get("script").getAsString());
    }

    @Override
    public String getScript(String indent) {
        if (!hasScript()) {
            return StringUtils.EMPTY;
        }

        String script = innerParam.get("script").getAsString();
        return LINE_SEPARATOR
                + Arrays.stream(script.split(LINE_SEPARATOR)).map(line -> indent + line).collect(Collectors.joining(LINE_SEPARATOR))
                + LINE_SEPARATOR;
    }

    public Optional<KerasLayers> getInnerLayer() {
        return Optional.ofNullable(this.innerLayer);
    }
}
