package com.samsung.sds.brightics.server.service;

import com.google.gson.JsonObject;
import com.google.protobuf.InvalidProtocolBufferException;
import com.samsung.sds.brightics.common.core.exception.AbsBrighticsException;
import com.samsung.sds.brightics.common.core.exception.BrighticsCoreException;
import com.samsung.sds.brightics.common.core.exception.BrighticsUncodedException;
import com.samsung.sds.brightics.common.core.util.IdGenerator;
import com.samsung.sds.brightics.common.core.util.JsonUtil;
import com.samsung.sds.brightics.common.network.proto.FailResult;
import com.samsung.sds.brightics.common.network.proto.MessageStatus;
import com.samsung.sds.brightics.common.network.proto.SuccessResult;
import com.samsung.sds.brightics.common.network.proto.deeplearning.ExecuteDLMessage;
import com.samsung.sds.brightics.common.network.proto.deeplearning.ExecuteDLMessage.DLActionType;
import com.samsung.sds.brightics.common.network.proto.deeplearning.ResultDLMessage;
import com.samsung.sds.brightics.common.network.util.ParameterBuilder;
import com.samsung.sds.brightics.common.workflow.flowrunner.vo.JobParam;
import com.samsung.sds.brightics.server.common.message.MessageManagerProvider;
import com.samsung.sds.brightics.server.common.message.task.TaskMessageBuilder;
import com.samsung.sds.brightics.server.common.message.task.TaskMessageRepository;
import com.samsung.sds.brightics.server.common.util.AuthenticationUtil;
import com.samsung.sds.brightics.server.common.util.keras.KerasModelScriptGenerator;
import com.samsung.sds.brightics.server.common.util.keras.generator.KerasExportScriptGenerator;
import com.samsung.sds.brightics.server.common.util.keras.generator.KerasSummaryScriptGenerator;
import com.samsung.sds.brightics.server.common.util.keras.generator.KerasTrainScriptGenerator;
import com.samsung.sds.brightics.server.model.entity.BrtcJobStatusError;
import com.samsung.sds.brightics.server.model.vo.ExceptionInfoVO;
import com.samsung.sds.brightics.server.service.repository.JobRepository;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.file.Paths;
import java.util.*;

@Service
public class DeeplearningService {

	private static final Logger logger = LoggerFactory.getLogger(DeeplearningService.class);

	@Autowired
	private AgentUserService agentUserService;

	@Autowired
    TaskService taskService;

	@Autowired
    JobRepository jobRepository;

	@Autowired
	private AgentService agentService;

	@Autowired
	private JobStatusService jobStatusService;

	@Autowired
	private MessageManagerProvider messageManager;

	@Value("${brightics.dl.repo.path:./dl}")
	private String DL_HOME;

	private static final String LOG_PATH = "/log/";

	public void executeDLScript(JsonObject model, String jid) {
		try {
			String taskId = IdGenerator.getSimpleId();
			List<String> taskList = new ArrayList<>();
			taskList.add(taskId);
			jobRepository.saveDLTaskIdsAsJobId(jid, taskList);

			KerasModelScriptGenerator generator = new KerasTrainScriptGenerator(model, jid);
			String kerasScript = generator.getFunctionalScript();
			String param = ParameterBuilder.newBuild().addProperty("logPath", DL_HOME + LOG_PATH)
					.addProperty("logName", jid).addProperty("script", kerasScript)
					.build();
			String attr = ParameterBuilder.newBuild().addProperty("mid", model.get("mid").getAsString()).build();
			taskService
					.executeTask(TaskMessageBuilder.newBuilder(taskId, "DLPythonScript").setParameters(param).setAttributes(attr).build());
		} catch (AbsBrighticsException e) {
			logger.error("[DL]", e);
			throw e;
		} catch (Exception e) {
			logger.error("[DL]", e);
			throw new BrighticsCoreException("3134", e.getMessage() + ".");
		}
	}

	public Object modelCheck(JobParam jobParam) {
		agentService.initAgent(agentUserService.getAgentIdAsUserId(jobParam.getUser()));
		try {
			String jid = jobParam.getJid();
			String main = jobParam.getMain();
			JsonObject models = JsonUtil.toJsonObject(jobParam).getAsJsonObject("models");
			JsonObject model = models.get(main).getAsJsonObject();
			model.addProperty("user", jobParam.getUser());
			KerasModelScriptGenerator generator = new KerasSummaryScriptGenerator(model, jid);
			String kerasScript = generator.getFunctionalScript();
			String taskId = IdGenerator.getSimpleId();

			String param = ParameterBuilder.newBuild().addProperty("logPath", DL_HOME + LOG_PATH)
					.addProperty("logName", jid).addProperty("script", kerasScript).build();
			String attr = ParameterBuilder.newBuild().addProperty("mid", model.get("mid").getAsString()).build();

			taskService
					.executeTask(TaskMessageBuilder.newBuilder(taskId, "DLPythonScript").setParameters(param).setAttributes(attr).build());
			while (!TaskMessageRepository.isExistFinishMessage(taskId)) {
				Thread.sleep(50L);
			}
			messageManager.taskManager().getAsyncTaskResult(taskId);
			return getDLStatusAsAgent(jid);
		} catch (AbsBrighticsException e) {
			logger.error("[DL]", e);
			return formatDLFailResult(jobParam.getJid(), e.getMessage(), ExceptionUtils.getStackTrace(e));
		} catch (Exception e) {
			logger.error("[DL]", e);
			return formatDLFailResult(jobParam.getJid(),
					(new BrighticsCoreException("3135", e.getMessage())).getMessage(), ExceptionUtils.getStackTrace(e));
		}
	}

	public String exportDLScript(JobParam jobParam) {
		try {
			JsonObject models = JsonUtil.toJsonObject(jobParam).getAsJsonObject("models");
			JsonObject model = models.get(jobParam.getMain()).getAsJsonObject();
			model.addProperty("user", jobParam.getUser());
			KerasModelScriptGenerator generator = new KerasExportScriptGenerator(model);
			return generator.getFunctionalScript();
		} catch (Exception e) {
			logger.error("[DL]", e);
			throw new BrighticsCoreException("3133", e.getMessage());
		}
	}

	public Object getDLStatus(String userId, String main) {
		List<Map<String, String>> jobListAsUserAndMain = jobStatusService.getJobListByUserAndMain(userId, main);
		if (jobListAsUserAndMain == null || jobListAsUserAndMain.isEmpty()) {
			return new HashMap<String, Object>();
		}
		
		Optional<Map<String, String>> lastDLInfoOption = jobListAsUserAndMain.stream()
		.filter(jobInfo -> jobInfo.get("modelType").equals("deeplearning")).reduce((first, second) -> second);
		if(lastDLInfoOption.isPresent()){
			Map<String, String> lastDLInfo = lastDLInfoOption.get();
			String jid = lastDLInfo.get("jid");
			if(!jobRepository.isContainDLJob(jid)) {
				// terminated job
				return formatDLFailResult(jid,
						(new BrighticsCoreException("4326")).getMessage(), null);
			} else if (lastDLInfo.get("status").equals(JobRepository.STATE_FAIL)) {
				jobRepository.removeDLJob(jid);
				BrtcJobStatusError jobstatusErrorInfo = jobStatusService.getJobStatusErrorInfo(jid);
				return formatDLFailResult(jid,
						(new BrighticsCoreException("3133", jobstatusErrorInfo.getMessage())).getMessage(), null);
			} else {
				return getDLStatusAsAgent(jid);
			}
		} else {
			return new HashMap<String, Object>();
		}
		
	}

	private Object getDLStatusAsAgent(String jid) {
		try {
			String param = ParameterBuilder.newBuild().addProperty("logPath", DL_HOME + LOG_PATH).addProperty("logName", jid).build();
			ExecuteDLMessage message = ExecuteDLMessage.newBuilder().setActionType(DLActionType.STATUS).setUser(AuthenticationUtil.getRequestUserId()).setParameters(param).build();
			ResultDLMessage result = messageManager.deeplearningManager().sendDeeplearningInfo(message);
			Map<String, Object> status = JsonUtil.jsonToMap(deeplearningResultParser(result));
			if (JobRepository.STATE_SUCCESS.equals(status.get("status").toString())) {
				jobRepository.removeDLJob(jid);
			}
			status.put("jid", jid);
			return status;
		} catch (AbsBrighticsException e) {
			logger.error("[DL] {}", e.detailedCause);
			jobRepository.removeDLJob(jid);
			return formatDLFailResult(jid, e.getMessage(), e.detailedCause);
		} catch (Exception e) {
			logger.error("[DL] {}", e.getMessage());
			jobRepository.removeDLJob(jid);
			return formatDLFailResult(jid, (new BrighticsCoreException("3135", e.getMessage())).getMessage(),
					ExceptionUtils.getStackTrace(e));
		}
	}

	public Object dlFileBrowse(String path) {
		String param = ParameterBuilder.newBuild().addProperty("path", Paths.get(DL_HOME, path).toString()).build();
		ExecuteDLMessage message = ExecuteDLMessage.newBuilder().setActionType(DLActionType.BROWSE).setUser(AuthenticationUtil.getRequestUserId()).setParameters(param).build();
		ResultDLMessage result = messageManager.deeplearningManager().sendDeeplearningInfo(message);
		return makePathRelative(deeplearningResultParser(result));
	}

	private Object makePathRelative(String result) {
		return result
				.replaceAll("\\\\\\\\", "/")
				.replaceAll("\"path\"[\\s]*:[\\s]*\"" + DL_HOME, "\"path\":\"");
	}

	private Object formatDLFailResult(String jid, String message, String detailMessage) {
		Map<String, Object> resultMap = new HashMap<>();
		resultMap.put("jid", jid);
		resultMap.put("status", JobRepository.STATE_FAIL);
		List<ExceptionInfoVO> errorInfoList = new ArrayList<>();
		errorInfoList.add(new ExceptionInfoVO(message, detailMessage));
		resultMap.put("errorInfo", errorInfoList);
		return resultMap;
	}
	
	
	private String deeplearningResultParser(ResultDLMessage result) {
		try {
			if (result.getMessageStatus() == MessageStatus.SUCCESS) {
				return result.getResult().unpack(SuccessResult.class).getResult();
			} else {
				FailResult failResult;
				failResult = result.getResult().unpack(FailResult.class);
				throw new BrighticsUncodedException(failResult.getMessage(), failResult.getDetailMessage());
			}
		} catch (InvalidProtocolBufferException e) {
			logger.error("Cannot parse deeplearning result.", e);
			throw new BrighticsCoreException("3001").addDetailMessage(ExceptionUtils.getStackTrace(e));
		}
	}

}
