# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

try:
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    __all__ = ["OpenTSLM"]
except ImportError:
    # open_flamingo is an optional dep; ahri / PhysicsTSLM do not need it.
    __all__ = []