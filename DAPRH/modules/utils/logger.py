import sys
from loguru import logger
import numpy as np
def statistic(scores, name="haha", logger=None):
    silhouettes_g = scores
    logger.log(f"======{name} score=======")
    logger.log("\t--Mean: {}".format(silhouettes_g.mean()))
    logger.log("\t--Quantile=0.2    :{} ".format(np.quantile(silhouettes_g, 0.2)))
    logger.log("\t--Quantile=0.15   :{} ".format(np.quantile(silhouettes_g, 0.15)))
    logger.log("\t--Quantile=0.1    :{} ".format(np.quantile(silhouettes_g, 0.1)))
    logger.log("\t--Quantile=0.08   :{} ".format(np.quantile(silhouettes_g, 0.08)))
    logger.log("\t--Quantile=0.05   :{} ".format(np.quantile(silhouettes_g, 0.05)))
    logger.log("\t--Quantile=0.01   :{} ".format(np.quantile(silhouettes_g, 0.01)))
    logger.log("\t--Quantile=0.005   :{} ".format(np.quantile(silhouettes_g, 0.005)))


class Logger(object):
    def __init__(self) -> None:
        #formatlog
        self.format={
            "TRAINLOG":"<red>(NDA)</red>|<blue><b>[TRAINGLOG]</b></blue>|<fg #AF5FD7>{message}</fg #AF5FD7>",
            "TESTLOG" :"<red>(NDA)</red>|<cyan><b>[TESTLOG  ]</b></cyan>|<fg #ffccb3>{message}</fg #ffccb3>",

            "INFO"   : "<red>(NDA)</red>|<green>{time:HH:mm:ss}</green>| --> <yellow>{message}</yellow>"
        }

        #other level
        logger.level("TRAINLOG", no=40, color="<yellow>", icon="🐍")
        logger.level("TESTLOG", no=41, color="<blue>", icon="🐖")
        self.logger = logger

    def __trigger(self,level="INFO", msg=""):
        '''Args
        level: (str) type of logs channel/process. examples: TRAINLOG, TESLOG
        msg  : (str) message
        '''
        level = level.upper()
        logger.remove()
        logger.add(sys.stdout, format=self.format[level], level=level, colorize=True)
        self.logger.log(level, msg)

    def traininglog(self, epoch, i, iters, avgloss, **kwargs):
        msg = f"[Epoch {epoch}:{i}/{iters}]:   --SumLoss: {avgloss:.4f}"
        for k in kwargs:  msg += "  --{}: {}".format(k, kwargs[k])
        self.__trigger("trainlog", msg)

    def validatinglog(self, mAP=None, top_k=None , cmc_topk=None, **kwargs):
        msg = "  Mean AP: {:4.1%}".format(mAP)
        self.__trigger("testlog", msg)
        if top_k is None or cmc_topk is None : return
        for k in cmc_topk:
            msg = '  top-{:<4}:{:6.3%}'.format(k, top_k[k-1])
            self.__trigger("testlog", msg)

    def log(self, msg):
        self.__trigger("info", msg=msg)


if __name__ == "__main__":
    l = Logger()
    l.log("trainlog", "cuu be voi !!!!")
    l.log("testlog", "okbaby!!!")
    l.log("trainlog", "Lo".format())