package com.samsung.sds.brightics.server.common.holder;

import com.samsung.sds.brightics.server.service.*;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.stereotype.Component;
import org.springframework.util.Assert;

import com.samsung.sds.brightics.server.common.message.MessageManagerProvider;

@Component
public final class BeanHolder implements ApplicationContextAware {

	@Autowired
	public JobStatusService jobStatusService;

	@Autowired
	public DataSourceService dataSourceService;

	@Autowired
	public TaskService taskService;

    @Autowired
    public DeeplearningService deeplearningService;

	@Autowired
	public MessageManagerProvider messageManager;

	@Autowired
	public DataService dataService;

	@Autowired
	public PyFunctionService pyFunctionService;

	@Autowired
	public MetadataConverterService metadataConverterService;

	private BeanHolder() {

	}

	private static final ThreadLocal<BeanHolder> serviceBeanHolder = new InheritableThreadLocal<>();

	@Override
	public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
		serviceBeanHolder.set(applicationContext.getBean(this.getClass()));
	}

	public static BeanHolder getBeanHolder() {
		Assert.notNull(serviceBeanHolder.get(), "beanHolder has not been initialized");
		return serviceBeanHolder.get();
	}

}
