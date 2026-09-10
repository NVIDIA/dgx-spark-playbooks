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
import { NextRequest, NextResponse } from 'next/server';
import { getGraphDbService } from '@/lib/graph-db-util';
import { getGraphDbType } from '../../settings/route';

/**
 * API endpoint for disconnecting from the selected graph database
 * POST /api/graph-db/disconnect
 */
export async function POST(request: NextRequest) {
  try {
    // Get the graph database type from the settings
    const graphDbType = getGraphDbType();
    console.log(`Disconnecting from ${graphDbType}...`);
    
    // Get the appropriate service
    const graphDbService = getGraphDbService(graphDbType);
    
    if (graphDbService.isInitialized()) {
      graphDbService.close();
      return NextResponse.json({
        success: true,
        message: `Successfully disconnected from ${graphDbType}`,
        type: graphDbType
      });
    } else {
      return NextResponse.json({
        success: false,
        message: `No active ${graphDbType} connection to disconnect`,
        type: graphDbType
      });
    }
  } catch (error) {
    console.error('Error disconnecting from graph database:', error);
    return NextResponse.json(
      { 
        error: `Failed to disconnect from graph database: ${error instanceof Error ? error.message : String(error)}`,
        type: getGraphDbType()
      },
      { status: 500 }
    );
  }
} 