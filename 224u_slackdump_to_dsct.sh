#!/bin/sh
#
# Usage:      224u_slackdump_to_dsct.sh <slack-dump.json>
#
jq -c '.messages[] | {conversation_id:(.ts), turns:([.,(.slackdump_thread_replies[]?)] | sort_by(.ts) | map({utterance:.text, theme_label:null, utterance_id:.client_msg_id}))}' $1
