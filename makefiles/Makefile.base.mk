# Copyright 2023 Autonomi AI, Inc. All rights reserved.

help-base:
	@echo "⚡️ NOS - Nitrous Oxide for your AI Infrastructure"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  docker-build-cpu    Build CPU docker image"
	@echo "  docker-build-gpu    Build GPU docker image"
	@echo "  docker-build-all    Build CPU and GPU docker images"
	@echo "  docker-test-cpu     Test CPU docker image"
	@echo "  docker-test-gpu     Test GPU docker image"
	@echo "  docker-test-all     Test CPU and GPU docker images"
	@echo "  docker-compose-upd-gpu    Run GPU docker image"
	@echo "  docker-compose-upd-cpu    Run CPU docker image"
	@echo "  docker-push-cpu     Push CPU docker image"
	@echo "  docker-push-gpu     Push GPU docker image"
	@echo "  docker-push-all     Push CPU and GPU docker images"
	@echo ""

.docker-build-and-push-multiplatform:
	@echo "🛠️ Building docker image"
	@echo "BASE_IMAGE: ${BASE_IMAGE}"
	@echo "TARGET: ${TARGET}"
	@echo "DOCKER_TARGET: ${DOCKER_TARGET}"
	@echo "IMAGE: ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET}"
	@echo "docker buildx create --use"
	docker buildx build -f docker/Dockerfile.multiplatform.${TARGET} \
		--platform linux/amd64,linux/arm64 \
		--target ${DOCKER_TARGET} \
		-t ${DOCKER_IMAGE_NAME}:latest-${TARGET} \
		-t ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET} \
		.

.docker-run:
	docker run -it ${DOCKER_ARGS} --rm \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET} \
		${DOCKER_CMD}

.docker-push-base:
	docker push ${DOCKER_IMAGE_NAME}:latest-${TARGET}
	docker push ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET}

docker-build-cpu: agi-build-cpu
docker-build-cpu-prod:
	make agi-build-cpu AGIPACK_ARGS=--prod

docker-build-gpu: agi-build-gpu
docker-build-gpu-prod:
	make agi-build-gpu AGIPACK_ARGS=--prod

docker-build-cu118: agi-build-cu118

docker-build-and-push-multiplatform-cpu:
	agi-pack generate ${AGIPACK_ARGS} \
		-c docker/agibuild.cpu.yaml \
		-o docker/Dockerfile.multiplatform.cpu \
		-p 3.8.15 \
		-b debian:buster-slim \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-{target}'
	make .docker-build-and-push-multiplatform \
		TARGET=cpu DOCKER_TARGET=cpu
docker-build-and-push-multiplatform-cpu-prod:
	make .docker-build-and-push-multiplatform-cpu AGIPACK_ARGS=--prod

docker-build-all: \
	docker-build-cpu docker-build-gpu docker-build-cu118

docker-push-cpu:
	make .docker-push-base \
	TARGET=cpu

docker-push-gpu:
	make .docker-push-base \
	TARGET=gpu

docker-push-cu118: agi-push-cu118

docker-push-all: \
	docker-push-cpu docker-push-gpu docker-push-cu118

docker-test-cpu:
	docker compose -f docker-compose.test.yml run --rm --build test-cpu

docker-test-gpu:
	docker compose -f docker-compose.test.yml run --rm --build test-gpu
