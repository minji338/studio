package com.samsung.sds.brightics.agent.service;

import com.google.protobuf.Any;
import com.samsung.sds.brightics.common.core.exception.AbsBrighticsException;
import com.samsung.sds.brightics.common.core.exception.BrighticsCoreException;
import com.samsung.sds.brightics.common.core.util.JsonUtil;
import com.samsung.sds.brightics.common.core.util.JsonUtil.JsonParam;
import com.samsung.sds.brightics.common.data.client.LocalFileClient;
import com.samsung.sds.brightics.common.network.proto.FailResult;
import com.samsung.sds.brightics.common.network.proto.FailResult.Builder;
import com.samsung.sds.brightics.common.network.proto.MessageStatus;
import com.samsung.sds.brightics.common.network.proto.SuccessResult;
import com.samsung.sds.brightics.common.network.proto.deeplearning.ExecuteDLMessage;
import com.samsung.sds.brightics.common.network.proto.deeplearning.ExecuteDLMessage.DLActionType;
import com.samsung.sds.brightics.common.network.proto.deeplearning.ResultDLMessage;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class DeeplearningService {

    private static final Logger logger = LoggerFactory.getLogger(DeeplearningService.class);

    public static ResultDLMessage manipulateDeeplearning(ExecuteDLMessage request) {
        try {
            if (request.getActionType() == DLActionType.BROWSE) {
                return browseDLFile(request);
            } else if (request.getActionType() == DLActionType.STATUS) {
                return getDLStatus(request);
            } else {
                logger.error("cannot manipulate deeplearning.");
                return getFailResult(new BrighticsCoreException("3002", "deeplearning " + request.getActionType()));
            }
        } catch (AbsBrighticsException e) {
            logger.error("cannot manipulate deeplearning.");
            return getFailResult(e);
        } catch (Exception e) {
            logger.error("cannot manipulate deeplearning.");
            return getFailResult(new BrighticsCoreException("4406").addDetailMessage(ExceptionUtils.getStackTrace(e)));
        }
    }

    /**
     * Get deep learning dynamic status. (error / model check / result log)
     */
    private static ResultDLMessage getDLStatus(ExecuteDLMessage request) {
        try {
            JsonParam jsonToParam = JsonUtil.jsonToParam(request.getParameters());
            String logPath = jsonToParam.getOrException("logPath", "3002", "log path");
            String logName = jsonToParam.getOrException("logName", "3002", "log name");
            String logFile = logPath + logName;
            File errorfile = new File(logFile + ".err");
            if (errorfile.exists()) {
                // get error message.
                try (BufferedReader errorfileReader = new BufferedReader(
                        new InputStreamReader(new FileInputStream(errorfile), "UTF-8"))) {
                    String errormessage = errorfileReader.lines().collect(Collectors.joining("\n"));
                    return getFailResult(new BrighticsCoreException("4402", errormessage));
                }
            } else {

                String status = "PROCESSING";
                Object resultColumnInfos = ArrayUtils.EMPTY_STRING_ARRAY;
                Object resultRows = ArrayUtils.EMPTY_STRING_ARRAY;

                File summaryfile = new File(logFile + ".summary");
                File logfile = new File(logFile + ".log");

                if (summaryfile.exists()) {
                    // get model check result.
                    try (BufferedReader summaryfileReader = new BufferedReader(
                            new InputStreamReader(new FileInputStream(summaryfile), "UTF-8"))) {
                        String summarymessage = summaryfileReader.lines().collect(Collectors.joining("\n"));
                        status = "SUCCESS";
                        resultColumnInfos = new String[]{"summary"};
                        resultRows = summarymessage;
                    }
                } else if (logfile.exists()) {
                    // get deep learning execute result.
                    try (BufferedReader logfileReader = new BufferedReader(
                            new InputStreamReader(new FileInputStream(logfile), "UTF-8"))) {
                        String columnInfoString = logfileReader.readLine();
                        List<Map<String, String>> columnInfos = new ArrayList<>();
                        for (String columnname : columnInfoString.split(",")) {
                            Map<String, String> columnInfo = new HashMap<>();
                            columnInfo.put("name", columnname);
                            columnInfos.add(columnInfo);
                        }
                        List<String> originRows = logfileReader.lines().filter(StringUtils::isNoneBlank)
                                .collect(Collectors.toList());
                        List<String[]> rows = new ArrayList<>();
                        for (String row : originRows) {
                            if ("END".equals(row)) {
                                status = "SUCCESS";
                                break;
                            } else {
                                rows.add(row.split(","));
                            }
                        }
                        resultColumnInfos = columnInfos;
                        resultRows = rows;
                    }
                }
                Map<String, Object> resultMap = new HashMap<>();
                resultMap.put("status", status);
                resultMap.put("columns", resultColumnInfos);
                resultMap.put("data", resultRows);
                return getSuccessResult(resultMap);

            }
        } catch (Exception e) {
            logger.error("cannot get DL status.", e);
            return getFailResult(
                    new BrighticsCoreException("4401", "").addDetailMessage(ExceptionUtils.getStackTrace(e)));
        }
    }

    private static ResultDLMessage browseDLFile(ExecuteDLMessage request) {
        try {
            JsonParam jsonToParam = JsonUtil.jsonToParam(request.getParameters());
            String path = jsonToParam.getOrException("path", "3002", "path");
            Object browse = LocalFileClient.browse(path);
            return getSuccessResult(browse);
        } catch (Exception e) {
            logger.error("cannot browse file.", e);
            return getFailResult(new BrighticsCoreException("4411").addDetailMessage(ExceptionUtils.getStackTrace(e)));
        }

    }

    private static ResultDLMessage getSuccessResult(Object result) {
        try {
            return getSuccessResult(JsonUtil.toJson(result));
        } catch (Exception e) {
            return getFailResult(new BrighticsCoreException("3102", "error in processing json")
                    .addDetailMessage(ExceptionUtils.getStackTrace(e)));
        }
    }

    private static ResultDLMessage getSuccessResult(String result) {
        SuccessResult build = SuccessResult.newBuilder().setResult(result).build();
        return ResultDLMessage.newBuilder().setMessageStatus(MessageStatus.SUCCESS).setResult(Any.pack(build)).build();
    }

    private static ResultDLMessage getFailResult(AbsBrighticsException e) {
        logger.error("[Meta data detail error message]", e);
        Builder failResult = FailResult.newBuilder().setMessage(e.getMessage());
        if (StringUtils.isNoneBlank(e.detailedCause)) {
            failResult.setDetailMessage(e.detailedCause);
        }
        return ResultDLMessage.newBuilder().setMessageStatus(MessageStatus.FAIL).setResult(Any.pack(failResult.build()))
                .build();
    }

}
