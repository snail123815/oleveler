import logging
import sys
from .version import infoText

logger = logging.getLogger()
logging.getLogger('matplotlib.font_manager').disabled = True
if len(logger.handlers) == 0:
    logFormatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] - %(message)s'
    )
    logfhandler = logging.FileHandler('dataProcessing.log')
    logfhandler.setLevel(logging.INFO)
    logfhandler.setFormatter(logFormatter)
    logshandler = logging.StreamHandler(stream=sys.stdout)
    logshandler.setLevel(logging.INFO)
    logshandler.setFormatter(logFormatter)
    logger.addHandler(logfhandler)
    logger.addHandler(logshandler)
    logger.setLevel(logging.INFO)

logger.info(infoText)
