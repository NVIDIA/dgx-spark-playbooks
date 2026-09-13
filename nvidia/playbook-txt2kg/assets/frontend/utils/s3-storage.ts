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
import { S3Client, PutObjectCommand, GetObjectCommand, ListObjectsV2Command, DeleteObjectCommand, _Object } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

// Get environment variables with defaults
const getS3Config = () => {
  return {
    endpoint: process.env.S3_ENDPOINT || 'http://localhost:9000',
    region: process.env.S3_REGION || 'us-east-1',
    bucket: process.env.S3_BUCKET || 'txt2kg',
    credentials: {
      accessKeyId: process.env.S3_ACCESS_KEY || 'minioadmin',
      secretAccessKey: process.env.S3_SECRET_KEY || 'minioadmin'
    },
    forcePathStyle: true // Needed for MinIO and other S3-compatible storages
  };
};

// Initialize S3 client
const getS3Client = () => {
  const config = getS3Config();
  return new S3Client({
    endpoint: config.endpoint,
    region: config.region,
    credentials: config.credentials,
    forcePathStyle: config.forcePathStyle
  });
};

// Upload file to S3
export const uploadFileToS3 = async (file: File, path: string = ''): Promise<string> => {
  const s3Client = getS3Client();
  const config = getS3Config();
  
  // Create a readable stream from file
  const buffer = await file.arrayBuffer();
  
  // Create key path
  const key = path ? `${path}/${file.name}` : file.name;
  
  const params = {
    Bucket: config.bucket,
    Key: key,
    Body: Buffer.from(buffer),
    ContentType: file.type || 'application/octet-stream'
  };
  
  try {
    await s3Client.send(new PutObjectCommand(params));
    return key;
  } catch (error) {
    console.error('Error uploading file to S3:', error);
    throw new Error(`Failed to upload file to S3: ${error instanceof Error ? error.message : String(error)}`);
  }
};

// Download file from S3
export const getFileFromS3 = async (key: string): Promise<Blob> => {
  const s3Client = getS3Client();
  const config = getS3Config();
  
  const params = {
    Bucket: config.bucket,
    Key: key
  };
  
  try {
    const response = await s3Client.send(new GetObjectCommand(params));
    const responseBody = await response.Body?.transformToByteArray();
    
    if (!responseBody) {
      throw new Error('Empty response body');
    }
    
    return new Blob([responseBody], { type: response.ContentType || 'application/octet-stream' });
  } catch (error) {
    console.error('Error downloading file from S3:', error);
    throw new Error(`Failed to download file from S3: ${error instanceof Error ? error.message : String(error)}`);
  }
};

// List files in S3 bucket
export const listFilesInS3 = async (prefix: string = ''): Promise<Array<{ key: string, size: number, lastModified: Date }>> => {
  const s3Client = getS3Client();
  const config = getS3Config();
  
  const params = {
    Bucket: config.bucket,
    Prefix: prefix
  };
  
  try {
    const response = await s3Client.send(new ListObjectsV2Command(params));
    return (response.Contents || []).map((item: _Object) => ({
      key: item.Key || '',
      size: item.Size || 0,
      lastModified: item.LastModified || new Date()
    }));
  } catch (error) {
    console.error('Error listing files in S3:', error);
    throw new Error(`Failed to list files in S3: ${error instanceof Error ? error.message : String(error)}`);
  }
};

// Delete file from S3
export const deleteFileFromS3 = async (key: string): Promise<void> => {
  const s3Client = getS3Client();
  const config = getS3Config();
  
  const params = {
    Bucket: config.bucket,
    Key: key
  };
  
  try {
    await s3Client.send(new DeleteObjectCommand(params));
  } catch (error) {
    console.error('Error deleting file from S3:', error);
    throw new Error(`Failed to delete file from S3: ${error instanceof Error ? error.message : String(error)}`);
  }
};

// Generate pre-signed URL for file download (useful for browser direct downloads)
export const getSignedUrlForS3File = async (key: string, expiresIn: number = 3600): Promise<string> => {
  const s3Client = getS3Client();
  const config = getS3Config();
  
  const params = {
    Bucket: config.bucket,
    Key: key
  };
  
  try {
    return await getSignedUrl(s3Client, new GetObjectCommand(params), { expiresIn });
  } catch (error) {
    console.error('Error generating signed URL for S3 file:', error);
    throw new Error(`Failed to generate signed URL: ${error instanceof Error ? error.message : String(error)}`);
  }
}; 