TEMPLATE_PATH := lib/template.html
RSS_TEMPLATE_PATH := lib/rss-template.xml

.PHONY: all html feed

all: html feed

html: $(patsubst %.md,%.html,$(shell find . -type f -name "*.md"))

%.html: %.md
	$(eval file_dir := $(dir $<))
	$(eval base_name := $(basename $(notdir $<)))
	$(eval relroot := $(shell python3 -c "import os.path; print(os.path.relpath('$(shell pwd)', '$(file_dir)'))")) 
	cd $(file_dir) && \
	pandoc -f markdown "$(base_name).md" --template="$(relroot)/${TEMPLATE_PATH}" \
		-o "$(base_name).html" --mathml --citeproc --csl "$(relroot)/assets/ieee.csl" \
		--metadata csspath="$(relroot)/assets/" \
		--resource-path="$(relroot)/assets" \
		--metadata year=$(shell date +%Y) \
		--metadata author="the internet" && \
	cd - 

feed: feed.xml

feed.xml: feed.yaml ${RSS_TEMPLATE_PATH}
	pandoc -M updated="$$(date '+%a, %d %b %Y %T %z')"\
		--metadata-file=feed.yaml \
		--template=${RSS_TEMPLATE_PATH} \
		-t html \
		-o feed.xml < /dev/null
