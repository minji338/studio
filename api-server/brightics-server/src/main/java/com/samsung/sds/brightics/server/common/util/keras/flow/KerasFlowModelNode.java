package com.samsung.sds.brightics.server.common.util.keras.flow;

import com.google.gson.JsonObject;
import org.apache.commons.lang3.StringUtils;

import java.util.Arrays;
import java.util.stream.Collectors;

public class KerasFlowModelNode extends KerasFlowNode {

    public KerasFlowModelNode(String fid, JsonObject function) {
        super(fid, function);
    }

    public boolean hasScript() {
        return containsStringParam("script");
    }

    public String getScript() {
        return getScript("");
    }

    public String getScript(String indent) {
        if (!hasScript()) {
            return StringUtils.EMPTY;
        }

        String script = param.get("script").getAsString();
        return LINE_SEPARATOR
                + Arrays.stream(script.split(LINE_SEPARATOR)).map(line -> indent + line).collect(Collectors.joining(LINE_SEPARATOR))
                + LINE_SEPARATOR;
    }
}
