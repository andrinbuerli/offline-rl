###########################
# HELP
###########################
include scripts/*.mk

###########################
# VARIABLES
###########################
ifeq ($(OS),Windows_NT)
	VENV := .venv/Scripts
else
	VENV := .venv/bin
endif

PARAMS := 

DATA_PATH := ../data

DIR :=

FILE := 


###########################
# Data
###########################

generate_dataset: ##@Data generate dataset
	@echo "Generating dataset..."
	$(VENV)/python scripts/generate_dataset.py --config-name $(FILE) $(PARAMS)

###########################
# Project UTILS
###########################
ls_data: ##@Utils list local data
	ls -l -a $(DATA_PATH)

setup: ##@Utils setup python virtual environment and install requirements with uv
	rm -rf .venv
	uv sync

print_env: ##@Utils print environment variables
	@echo "export VENV=$(VENV)"
	@echo "export DATA_PATH=$(DATA_PATH)"
	@echo "export DIR=$(DIR)"
	@echo "export FILE=$(FILE)"
	@echo "export PARAMS=$(PARAMS)"
	
x11: ##@Utils export display to local machine for X11 forwarding
	@LOCAL_MACHINE_IP_ADDRESS=$(shell echo $$SSH_CLIENT | awk '{print $$1}'); \
	if [ -z "$$LOCAL_MACHINE_IP_ADDRESS" ]; then \
		echo "No SSH_CLIENT detected, falling back to hostname."; \
		LOCAL_MACHINE_IP_ADDRESS=$(shell hostname -I | awk '{print $$1}'); \
	fi; \
	if [ -z "$$LOCAL_MACHINE_IP_ADDRESS" ]; then \
		echo "Could not detect local machine IP."; \
		exit 1; \
	fi; \
	echo "Detected local machine IP: $$LOCAL_MACHINE_IP_ADDRESS"; \
	export DISPLAY=$$LOCAL_MACHINE_IP_ADDRESS:0.0; \
	echo "export DISPLAY=$$DISPLAY"
