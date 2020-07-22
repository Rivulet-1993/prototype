
LOCAL_PATH := $(shell pwd)


VERSION := $(CI_COMMIT_TAG)
PROJECT := algorithm/prototype

ifeq ($(VERSION), )
	VERSION := $(shell docker run --privileged --rm -v $(LOCAL_PATH):/project -v ~/.ssh:/root/.ssh -v ~/.gitconfig:/root/.gitconfig registry.sensetime.com/spring-test/spring-ci bash main.sh current_tag)
endif

all: image

image:
	echo "VERSION="$(VERSION) > .env
	docker build --pull -t registry.sensetime.com/spring-test/$(PROJECT):$(VERSION) .
	docker push registry.sensetime.com/spring-test/$(PROJECT):$(VERSION)
	docker tag registry.sensetime.com/spring-test/$(PROJECT):$(VERSION) registry.sensetime.com/spring-test/$(PROJECT):latest
	docker push registry.sensetime.com/spring-test/$(PROJECT):latest

tag:
	docker run --privileged --rm -v $(LOCAL_PATH):/project -v ~/.ssh:/root/.ssh -v ~/.gitconfig:/root/.gitconfig registry.sensetime.com/spring-test/spring-ci bash main.sh make_tag
