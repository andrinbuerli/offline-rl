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
# Train
###########################

train: ##@Train train the iql model
	@echo "Training the iql model..."
	@if [ -n "$(FILE)" ]; then \
		WANDB_API_KEY=$(shell cat .wandbkey) $(VENV)/python scripts/train.py --config-name $(FILE) $(PARAMS); \
	else \
		WANDB_API_KEY=$(shell cat .wandbkey) $(VENV)/python scripts/train.py $(PARAMS); \
	fi


###########################
# Eval
###########################

eval: ##@Train train the iql model
	@echo "Evaluating the iql model..."
	@if [ -n "$(FILE)" ]; then \
		WANDB_API_KEY=$(shell cat .wandbkey) $(VENV)/python scripts/eval.py --config-name $(FILE) $(PARAMS); \
	else \
		WANDB_API_KEY=$(shell cat .wandbkey) $(VENV)/python scripts/eval.py $(PARAMS); \
	fi

###########################
# Data
###########################

generate_dataset: ##@Data generate dataset
	@echo "Generating dataset..."
	@if [ -n "$(FILE)" ]; then \
		$(VENV)/python scripts/generate_dataset.py --config-name $(FILE) $(PARAMS); \
	else \
		$(VENV)/python scripts/generate_dataset.py $(PARAMS); \
	fi

###########################
# Project UTILS
###########################
ls_data: ##@Utils list local data
	minari list local

setup: ##@Utils setup python virtual environment and install requirements with uv
	rm -rf .venv
	uv sync

print_env: ##@Utils print environment variables
	@echo "export VENV=$(VENV)"
	@echo "export DATA_PATH=$(DATA_PATH)"
	@echo "export DIR=$(DIR)"
	@echo "export FILE=$(FILE)"
	@echo "export PARAMS=$(PARAMS)"
	@echo "export WANDB_API_KEY=$(shell cat .wandbkey)"
	
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
