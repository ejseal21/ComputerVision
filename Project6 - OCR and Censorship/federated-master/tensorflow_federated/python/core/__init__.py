# Lint as: python2
# Copyright 2018, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorFlow Federated core library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TensorFlow Federated uses Python imports to manage namespaces in ways that are
# different from the Google Python Style guide.
#
# pylint: disable=g-bad-import-order,wildcard-import
from tensorflow_federated.python.core.api import *

# NOTE: This import must happen after the wildcard import.
from tensorflow_federated.python.core import framework
from tensorflow_federated.python.core import utils
# pylint: enable=g-bad-import-order,wildcard-import