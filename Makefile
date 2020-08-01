
LOCAL_PATH := $(shell pwd)


VERSION := $(CI_COMMIT_TAG)
PROJECT := prototype
DOCKER_REPO := registry.sensetime.com/spring/ce/elements

ifeq ($(VERSION), )
	VERSION := $(shell docker run --privileged --rm -v $(LOCAL_PATH):/project -v ~/.ssh:/root/.ssh -v ~/.gitconfig:/root/.gitconfig registry.sensetime.com/spring-test/spring-ci bash main.sh current_tag)
endif

all: image

image:
	echo "VERSION="$(VERSION) > .env
	docker build --pull --network=host -t $(DOCKER_REPO)/$(PROJECT):$(VERSION) .
	docker push $(DOCKER_REPO)/$(PROJECT):$(VERSION)
	docker tag $(DOCKER_REPO)/$(PROJECT):$(VERSION) $(DOCKER_REPO)/$(PROJECT):latest
	docker push $(DOCKER_REPO)/$(PROJECT):latest

tag:
	docker run --privileged --rm -v $(LOCAL_PATH):/project -v ~/.ssh:/root/.ssh -v ~/.gitconfig:/root/.gitconfig registry.sensetime.com/spring-test/spring-ci bash main.sh make_tag
