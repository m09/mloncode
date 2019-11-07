check:
	black --check mloncode setup.py
	mypy mloncode setup.py
	flake8 --count mloncode setup.py
	pylint mloncode setup.py

bblfshd:
	docker start mloncode_bblfshd > /dev/null 2>&1 \
		|| docker run \
			--detach \
			--rm \
			--name mloncode_bblfshd \
			--privileged \
			--publish 9432:9432 \
			bblfsh/bblfshd:v2.14.0-drivers \
			--log-level DEBUG

.PHONY: check bblfshd