package com.samsung.sds.brightics.server.controller;

import com.samsung.sds.brightics.common.workflow.flowrunner.vo.JobParam;
import com.samsung.sds.brightics.server.service.DeeplearningService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;

@RestController
@RequestMapping("/api/core/v2/dl")
public class DeeplearningController {
    
    @Autowired
    private DeeplearningService deeplearningService;

    /**
     *  POST   /api/core/v2/dl/exportscript               : export DL script. 
     *  POST   /api/core/v2/dl/modelcheck                 : DL model valid check. 
     *  GET    /api/core/v2/dl/status/{userId}/{main}     : get last DL job. 
     *  GET    /api/core/v2/dl/browse				      : browse dl folder. 
     */
    
    @RequestMapping(value = "/exportscript", method = RequestMethod.POST)
    public Object getExportDLScript(@Valid @RequestBody JobParam jobParam) {
        return deeplearningService.exportDLScript(jobParam);
    }

    @RequestMapping(value = "/modelcheck", method = RequestMethod.POST)
    public Object modelcheck(@Valid @RequestBody JobParam jobParam) {
        return deeplearningService.modelCheck(jobParam);
    }
    
    @RequestMapping(value = "/status/{userId:.+}/{main}", method = RequestMethod.GET)
    public Object getJobstatusInfo(@PathVariable String userId, @PathVariable String main) {
        return deeplearningService.getDLStatus(userId, main);
    }

    @RequestMapping(value = "/browse", method = RequestMethod.GET)
    public Object dlFileBrowse(@RequestParam String path){
        return deeplearningService.dlFileBrowse(path);
    }
    
}
