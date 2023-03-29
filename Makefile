#-------------------------------------------------
# Project
#-------------------------------------------------
SRC_PATH		:= .
TARGET_NAME		:= Mobilint_Compiler_qb_User_Guide

SRCS			:= \
	$(wildcard $(SRC_PATH)/src/*) \
	$(wildcard $(SRC_PATH)/media/*) \
	$(wildcard ../../bin/codegen/*.docx) \
	$(SRC_PATH)/*.lua
	

#-------------------------------------------------
# Build commands
#-------------------------------------------------
BUILD_TARGET		:= $(TARGET_NAME).docx
BUILD_TARGET_PDF	:= $(TARGET_NAME).pdf

all: $(BUILD_TARGET)
	@echo Document Generation is done!
	@explorer $(BUILD_TARGET_PDF) &

clean:
	@rm -f $(BUILD_TARGET)
	@rm -f $(BUILD_TARGET_PDF)
	@echo Done!

$(BUILD_TARGET):$(SRCS)
	@echo '*** DocGen Build *** (Please wait a seconds!!!)'
	@codegen docgen -t mobilint main.lua $(BUILD_TARGET)
