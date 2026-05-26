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
import queryLoggerService, { QueryLogEntry } from '@/lib/query-logger';
import fs from 'fs';
import path from 'path';
import { promises as fsPromises } from 'fs';

interface QueryLogData {
  query: string;
  count: number;
}

interface FixResults {
  fixed: number;
  data: QueryLogData[];
}

/**
 * API endpoint to check and fix query logs
 */
export async function GET(request: NextRequest) {
  try {
    console.log('Checking and fixing query logs');
    
    // Initialize logger if not already
    if (!queryLoggerService.isInitialized()) {
      await queryLoggerService.initialize();
    }
    
    let results: FixResults = { fixed: 0, data: [] };
    
    try {
      // Get the log file path
      const logFilePath = path.join(process.cwd(), 'data', 'query-logs.json');
      
      // Check if log file exists
      if (!fs.existsSync(logFilePath)) {
        console.log('Log file does not exist, creating empty file');
        await fsPromises.mkdir(path.dirname(logFilePath), { recursive: true });
        await fsPromises.writeFile(logFilePath, JSON.stringify([]));
        return NextResponse.json({
          success: true,
          results,
          message: 'Created new empty log file'
        });
      }
      
      // Read existing logs
      const logsRaw = await fsPromises.readFile(logFilePath, 'utf-8');
      let logs: QueryLogEntry[] = JSON.parse(logsRaw || '[]');
      
      console.log(`Found ${logs.length} query log entries`);
      
      // Create a summary of existing logs
      const querySummary = new Map<string, number>();
      logs.forEach(log => {
        const count = querySummary.get(log.query) || 0;
        querySummary.set(log.query, count + 1);
      });
      
      // Convert to array for response
      results.data = Array.from(querySummary.entries()).map(([query, count]) => ({
        query,
        count
      }));
      
      // If there are no logs, add a default test log
      if (logs.length === 0) {
        console.log('No logs found, adding a default test log');
        
        const defaultLog: QueryLogEntry = {
          query: 'Test query for metrics',
          queryMode: 'traditional',
          timestamp: new Date().toISOString(),
          metrics: {
            executionTimeMs: 0,
            relevanceScore: 0,
            precision: 0,
            recall: 0,
            resultCount: 0
          }
        };
        
        logs.push(defaultLog);
        results.fixed++;
        
        // Update results data
        results.data.push({
          query: defaultLog.query,
          count: 1
        });
        
        // Write back to file
        await fsPromises.writeFile(logFilePath, JSON.stringify(logs, null, 2));
        console.log('Added default test log');
      }
      
      // Return the fixed results
      return NextResponse.json({
        success: true,
        results,
        message: `Fixed ${results.fixed} query logs`
      });
    } catch (error) {
      console.error('Error during fix operation:', error);
      return NextResponse.json({
        success: false,
        error: error instanceof Error ? error.message : String(error)
      }, { status: 500 });
    }
  } catch (error) {
    console.error('Error fixing query logs:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
} 