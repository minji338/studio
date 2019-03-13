package com.samsung.sds.brightics.agent.network.receiver;

import com.samsung.sds.brightics.agent.network.listener.ReceiveMessageListener;
import com.samsung.sds.brightics.agent.service.DeeplearningService;
import com.samsung.sds.brightics.common.core.thread.ThreadLocalContext;
import com.samsung.sds.brightics.common.network.proto.deeplearning.DeeplearningServiceGrpc;
import com.samsung.sds.brightics.common.network.proto.deeplearning.ExecuteDLMessage;
import com.samsung.sds.brightics.common.network.proto.deeplearning.ResultDLMessage;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DeeplearningReceiver extends DeeplearningServiceGrpc.DeeplearningServiceImplBase {

	private static final Logger logger = LoggerFactory.getLogger(DeeplearningReceiver.class);
	private static final String USER = "user";
	private final ReceiveMessageListener listener;

	public DeeplearningReceiver(ReceiveMessageListener listener) {
		this.listener = listener;
	}

	@Override
	public void receiveDeeplearningInfo(ExecuteDLMessage request, StreamObserver<ResultDLMessage> responseObserver) {
		String key = listener.receive(request);
		ThreadLocalContext.put(USER, request.getUser());
		logger.info("[Deeplearning receiver] receive deeplearning action : " + request.toString() + ", user : "
				+ request.getUser());

		ResultDLMessage resultMessage = DeeplearningService.manipulateDeeplearning(request);
		responseObserver.onNext(resultMessage);
		responseObserver.onCompleted();

		listener.onCompleted(key);
	}
}
