#!/bin/bash

# If there are new files with headers that can't match the conditions here,
# then the files can be ignored by an additional glob argument via the -g flag.
# For example:
#   -g "!src/special_file.rs"
#   -g "!src/special_directory"
#
# Accepted copyright lines (either Vello Authors for original files, or Ekrano Authors for new):
#   // Copyright YYYY the Vello Authors
#   // Copyright YYYY the Ekrano Authors
#   // Copyright YYYY the Vello Authors
#   // Copyright YYYY the Ekrano Authors  (both lines, for modified files)

# Check all the standard Rust source files
output=$(rg "^// Copyright (19|20)[\d]{2} (.+ and )?the (Vello|Ekrano) Authors( and .+)?$\n^// SPDX-License-Identifier: Apache-2\.0 OR MIT$\n\n" --files-without-match --multiline -g "*.rs" -g "!ekrano_shaders/src/cpu" .)

if [ -n "$output" ]; then
	echo -e "The following files lack the correct copyright header:\n"
	echo $output
	echo -e "\n\nFor new Ekrano files, please add:\n"
	echo "// Copyright $(date +%Y) the Ekrano Authors"
	echo "// SPDX-License-Identifier: Apache-2.0 OR MIT"
	echo -e "\nFor unmodified Vello files, the original header must be preserved:\n"
	echo "// Copyright YYYY the Vello Authors"
	echo "// SPDX-License-Identifier: Apache-2.0 OR MIT"
	echo -e "\n... rest of the file ...\n"
	exit 1
fi

# Check Slang sources and CPU shader Rust (Unlicense)
output=$(rg "^// Copyright (19|20)[\d]{2} (.+ and )?the (Vello|Ekrano) Authors( and .+)?$\n^// SPDX-License-Identifier: Apache-2\.0 OR MIT OR Unlicense$\n" --files-without-match --multiline -g "ekrano_shaders/{slang,src/cpu}/**/*.{rs,slang}" .)

if [ -n "$output" ]; then
        echo -e "The following shader files lack the correct copyright header:\n"
        echo $output
        echo -e "\n\nFor new Ekrano shaders, please add:\n"
        echo "// Copyright $(date +%Y) the Ekrano Authors"
        echo "// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense"
        echo -e "\n... rest of the file ...\n"
        exit 1
fi

echo "All files have correct copyright headers."
exit 0
