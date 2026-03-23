import sys
from src.logger.loggings import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: [{file_name}] line number: [{exc_tb.tb_lineno}] error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self):
        return f"{CustomException.__name__}({self.error_message!r})"
    
# Test the CustomException
if __name__ == "__main__":
    try:
        raise ValueError("This is a test error.")
    except Exception as e:
        logging.error(str(e))
        raise CustomException(e, sys)