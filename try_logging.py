import logging

# 创建 Logger
logger = logging.getLogger(__name__)
# 设置日志级别
logger.setLevel(logging.DEBUG)

# 创建 Handler（输出到控制台）
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# 创建 Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 将 Formatter 添加到 Handler
stream_handler.setFormatter(formatter)

# 将 Handler 添加到 Logger
logger.addHandler(stream_handler)

# 记录日志
logger.debug('这是一条调试信息')
logger.info('这是一条常规信息')
logger.warning('这是一条警告信息')
logger.error('这是一条错误信息')
logger.critical('这是一条严重错误信息')