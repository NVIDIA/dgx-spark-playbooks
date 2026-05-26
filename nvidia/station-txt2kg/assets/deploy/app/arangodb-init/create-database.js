//
// SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ArangoDB initialization script to create the txt2kg database
// This script is executed automatically when the ArangoDB container starts

db._createDatabase("txt2kg");
console.log("Database 'txt2kg' created successfully!");

// Optional: Create collections needed by your application
// Replace with actual collections you need
/*
const db = require("@arangodb").db;
db._useDatabase("txt2kg");

if (!db._collection("entities")) {
  db._createDocumentCollection("entities");
  console.log("Collection 'entities' created");
}

if (!db._collection("relationships")) {
  db._createEdgeCollection("relationships");
  console.log("Collection 'relationships' created");
}
*/ 