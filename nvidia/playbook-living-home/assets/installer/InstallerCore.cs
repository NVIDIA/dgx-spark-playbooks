using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Management;
using System.Net;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using Microsoft.Win32;

namespace LivingHome.Setup
{
    public sealed class SetupException : Exception
    {
        public SetupException(string message) : base(message) { }
    }

    public sealed class PayloadSpec
    {
        public string url { get; set; }
        public long size { get; set; }
        public string sha256 { get; set; }
        public long unpackedBytes { get; set; }
        public string entryPoint { get; set; }
    }
    public sealed class ModelSpec
    {
        public string path { get; set; }
        public string url { get; set; }
        public long size { get; set; }
        public string sha256 { get; set; }
    }
    public sealed class ReleaseManifest
    {
        public int schemaVersion { get; set; }
        public string releaseVersion { get; set; }
        public string architecture { get; set; }
        public PayloadSpec payload { get; set; }
        public List<ModelSpec> models { get; set; }
        public long minimumFreeBytes { get; set; }
    }
    public sealed class Source
    {
        public string FilePath;
        public Uri RemoteUri;
        public bool IsLocal { get { return FilePath != null; } }
        public string Display { get { return IsLocal ? FilePath : RemoteUri.GetLeftPart(UriPartial.Authority); } }
    }
    public sealed class LoadedRelease
    {
        public ReleaseManifest Manifest;
        public Source ManifestSource;
        public Source PayloadSource;
        public List<Source> ModelSources;
        public string ManifestHash;
        public bool IsPinned;
    }
    public sealed class ProgressUpdate
    {
        public string Message;
        public long Current;
        public long Total;
        public int Percent { get { return Total <= 0 ? 0 : (int)Math.Min(100, (double)Current * 100 / Total); } }
    }
    public sealed class CheckResult
    {
        public string name { get; set; }
        public string status { get; set; }
        public string detail { get; set; }
    }
    public sealed class ReadinessReport
    {
        public string checkedAtUtc { get; set; }
        public string releaseVersion { get; set; }
        public string manifestSha256 { get; set; }
        public bool manifestPinned { get; set; }
        public string installDirectory { get; set; }
        public bool canInstall { get; set; }
        public List<CheckResult> checks { get; set; }
    }

    public static class Safety
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern uint GetFileAttributes(string path);
        private static readonly Regex HashPattern = new Regex("^[a-fA-F0-9]{64}$", RegexOptions.CultureInvariant);
        private static readonly Regex DevicePattern = new Regex("^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\\..*)?$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        public static void ValidateHash(string value, string field)
        {
            if (value == null || !HashPattern.IsMatch(value)) throw new SetupException(field + " must contain exactly 64 hexadecimal SHA256 characters.");
        }
        public static string RelativePath(string value)
        {
            if (String.IsNullOrWhiteSpace(value) || value.Length > 220 || Path.IsPathRooted(value) || value.IndexOf(':') >= 0)
                throw new SetupException("Unsafe relative path: " + value);
            string normalized = value.Replace('/', '\\');
            string[] pieces = normalized.Split('\\');
            foreach (string piece in pieces)
            {
                if (String.IsNullOrEmpty(piece) || piece == "." || piece == ".." || piece.EndsWith(".") || piece.EndsWith(" ") ||
                    piece.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || piece.Length > 255 || DevicePattern.IsMatch(piece))
                    throw new SetupException("Unsafe relative path: " + value);
            }
            return normalized;
        }
        public static string Under(string root, string relative)
        {
            string prefix = Path.GetFullPath(root).TrimEnd('\\') + "\\";
            string result = Path.GetFullPath(Path.Combine(prefix, RelativePath(relative)));
            if (!result.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) throw new SetupException("Path escapes the installation directory.");
            return result;
        }
        public static void NoReparse(string path)
        {
            string absolute = Path.GetFullPath(path);
            string cursor = Path.GetPathRoot(absolute);
            if (cursor.StartsWith("\\\\", StringComparison.Ordinal)) throw new SetupException("Network file paths are unsupported. Choose a local drive.");
            foreach (string piece in absolute.Substring(cursor.Length).Split(new char[] { '\\' }, StringSplitOptions.RemoveEmptyEntries))
            {
                cursor = Path.Combine(cursor, piece);
                uint attributes = GetFileAttributes(cursor);
                if (attributes == 0xFFFFFFFF)
                {
                    int error = Marshal.GetLastWin32Error();
                    if (error != 2 && error != 3) throw new SetupException("Cannot verify path attributes (Windows error " + error + "): " + cursor);
                }
                else if ((attributes & (uint)FileAttributes.ReparsePoint) != 0)
                    throw new SetupException("A link, junction, or reparse point is not allowed in this path: " + cursor);
            }
        }
        public static string NewDestination(string path)
        {
            if (String.IsNullOrWhiteSpace(path) || !Path.IsPathRooted(path)) throw new SetupException("Choose an absolute installation directory on a local drive.");
            string full = Path.GetFullPath(path).TrimEnd('\\');
            if (full.Length > 140) throw new SetupException("Choose a shorter installation directory (at most 140 characters) so Windows can extract the payload safely.");
            if (full.IndexOf('%') >= 0 || full.IndexOf('!') >= 0) throw new SetupException("Choose an installation path without percent or exclamation characters; Windows batch entry points expand them.");
            if (String.Equals(full, Path.GetPathRoot(full).TrimEnd('\\'), StringComparison.OrdinalIgnoreCase)) throw new SetupException("A drive root cannot be an installation directory.");
            string leaf = Path.GetFileName(full);
            RelativePath(leaf);
            NoReparse(full);
            if (Directory.Exists(full) || File.Exists(full)) throw new SetupException("The installation directory already exists. Choose a new directory; existing installations are never overwritten.");
            string legacy = Path.GetFullPath(@"C:\LivingHome");
            if (full.Equals(legacy, StringComparison.OrdinalIgnoreCase) || full.StartsWith(legacy + "\\", StringComparison.OrdinalIgnoreCase))
                throw new SetupException("C:\\LivingHome is reserved for the existing workspace. Choose a separate installation directory.");
            string ancestor = Path.GetDirectoryName(full);
            if (!Directory.Exists(ancestor)) throw new SetupException("The parent directory does not exist. Choose an existing parent directory.");
            for (string current = ancestor; !String.IsNullOrEmpty(current); current = Path.GetDirectoryName(current))
            {
                if (File.Exists(Path.Combine(current, ".livinghome-install.json"))) throw new SetupException("The selected directory is inside an existing Living Home installation.");
                if (current == Path.GetPathRoot(current)) break;
            }
            return full;
        }
        public static void CreateDirectories(string root, string directory)
        {
            NoReparse(root);
            NoReparse(directory);
            string prefix = Path.GetFullPath(root).TrimEnd('\\') + "\\";
            if (!Path.GetFullPath(directory).StartsWith(prefix, StringComparison.OrdinalIgnoreCase) && !Path.GetFullPath(directory).Equals(Path.GetFullPath(root), StringComparison.OrdinalIgnoreCase))
                throw new SetupException("Directory escapes the staging root.");
            Directory.CreateDirectory(directory);
            NoReparse(directory);
        }
        public static void DeleteOwnedStaging(string directory, string marker)
        {
            NoReparse(directory);
            string ownership = Path.Combine(directory, ".setup-owner");
            if (!File.Exists(ownership) || File.ReadAllText(ownership) != marker) throw new SetupException("Staging ownership could not be verified; the directory was preserved.");
            var pending = new Stack<string>();
            pending.Push(directory);
            while (pending.Count > 0)
            {
                string current = pending.Pop();
                NoReparse(current);
                foreach (string item in Directory.GetFileSystemEntries(current))
                {
                    if ((File.GetAttributes(item) & FileAttributes.ReparsePoint) != 0) throw new SetupException("A reparse point was found during cleanup; the staging directory was preserved.");
                    if (Directory.Exists(item)) pending.Push(item);
                }
            }
            Directory.Delete(directory, true);
        }
    }

    public static class ManifestReader
    {
        private const int MaxManifestBytes = 1024 * 1024;
        public static Source ResolveManifest(string input)
        {
            if (String.IsNullOrWhiteSpace(input)) throw new SetupException("Choose a release manifest file or HTTPS URL.");
            Uri uri;
            if (Uri.TryCreate(input, UriKind.Absolute, out uri) && uri.Scheme == "https") return new Source { RemoteUri = uri };
            if (Uri.TryCreate(input, UriKind.Absolute, out uri) && uri.IsFile)
            {
                if (uri.IsUnc || !String.IsNullOrEmpty(uri.Host) && uri.Host != "localhost") throw new SetupException("Network manifest paths are unsupported; use HTTPS.");
                input = uri.LocalPath;
            }
            else if (Uri.TryCreate(input, UriKind.Absolute, out uri) && !uri.IsFile) throw new SetupException("Remote manifests must use HTTPS.");
            string full = Path.GetFullPath(input);
            Safety.NoReparse(full);
            return new Source { FilePath = full };
        }
        public static Source ResolveAsset(Source manifest, string input)
        {
            if (String.IsNullOrWhiteSpace(input)) throw new SetupException("Every payload and model needs a source URL or local file.");
            Uri uri;
            if (!manifest.IsLocal)
            {
                if (!Uri.TryCreate(manifest.RemoteUri, input, out uri) || uri.Scheme != "https" || !String.IsNullOrEmpty(uri.UserInfo))
                    throw new SetupException("Remote release assets must use HTTPS and cannot contain URL credentials.");
                return new Source { RemoteUri = uri };
            }
            if (Uri.TryCreate(input, UriKind.Absolute, out uri) && uri.Scheme == "https")
            {
                if (!String.IsNullOrEmpty(uri.UserInfo)) throw new SetupException("Asset URLs cannot contain URL credentials.");
                return new Source { RemoteUri = uri };
            }
            if (Uri.TryCreate(input, UriKind.Absolute, out uri) && uri.IsFile)
            {
                if (uri.IsUnc || !String.IsNullOrEmpty(uri.Host) && uri.Host != "localhost") throw new SetupException("Remote file shares are unsupported; use HTTPS.");
                input = uri.LocalPath;
            }
            else if (Uri.TryCreate(input, UriKind.Absolute, out uri) && !uri.IsFile) throw new SetupException("Remote release assets must use HTTPS.");
            string full = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(manifest.FilePath), input));
            Safety.NoReparse(full);
            return new Source { FilePath = full };
        }
        public static LoadedRelease Read(string input, string pin, CancellationToken cancel)
        {
            Source source = ResolveManifest(input);
            if (!source.IsLocal && !String.IsNullOrEmpty(source.RemoteUri.UserInfo)) throw new SetupException("Manifest URLs cannot contain URL credentials.");
            byte[] bytes;
            if (source.IsLocal)
            {
                if (!File.Exists(source.FilePath)) throw new SetupException("The manifest file does not exist.");
                using (var inputStream = new FileStream(source.FilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    if (inputStream.Length > MaxManifestBytes) throw new SetupException("The manifest exceeds the 1 MiB size limit.");
                    bytes = new byte[(int)inputStream.Length];
                    int read = 0, count;
                    while (read < bytes.Length && (count = inputStream.Read(bytes, read, bytes.Length - read)) > 0) { cancel.ThrowIfCancellationRequested(); read += count; }
                    if (read != bytes.Length) throw new SetupException("The manifest file is truncated.");
                }
            }
            else
            {
                using (HttpWebResponse response = Download.OpenResponse(source.RemoteUri, 0, cancel))
                using (Stream stream = response.GetResponseStream())
                using (var output = new MemoryStream())
                {
                    byte[] buffer = new byte[16384];
                    int count;
                    while ((count = stream.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        cancel.ThrowIfCancellationRequested();
                        if (output.Length + count > MaxManifestBytes) throw new SetupException("The manifest exceeds the 1 MiB size limit.");
                        output.Write(buffer, 0, count);
                    }
                    bytes = output.ToArray();
                }
            }
            string hash;
            using (var sha = SHA256.Create()) hash = Download.Hex(sha.ComputeHash(bytes));
            bool pinned = !String.IsNullOrWhiteSpace(pin);
            if (pinned)
            {
                Safety.ValidateHash(pin, "Manifest pin");
                if (!hash.Equals(pin, StringComparison.OrdinalIgnoreCase)) throw new SetupException("The release manifest SHA256 does not match the supplied pin. Obtain the manifest and pin from the release publisher.");
            }
            ReleaseManifest manifest;
            try
            {
                string json = new UTF8Encoding(false, true).GetString(bytes).TrimStart('\uFEFF');
                var serializer = new JavaScriptSerializer { MaxJsonLength = MaxManifestBytes, RecursionLimit = 30 };
                manifest = serializer.Deserialize<ReleaseManifest>(json);
            }
            catch (Exception ex) { throw new SetupException("The manifest is not valid UTF-8 release JSON: " + ex.Message); }
            Validate(manifest);
            var loaded = new LoadedRelease { Manifest = manifest, ManifestSource = source, ManifestHash = hash, IsPinned = pinned, ModelSources = new List<Source>() };
            loaded.PayloadSource = ResolveAsset(source, manifest.payload.url);
            foreach (ModelSpec model in manifest.models) loaded.ModelSources.Add(ResolveAsset(source, model.url));
            return loaded;
        }
        public static void Validate(ReleaseManifest manifest)
        {
            if (manifest == null || manifest.schemaVersion != 1) throw new SetupException("This installer requires manifest schemaVersion 1.");
            if (String.IsNullOrWhiteSpace(manifest.releaseVersion) || manifest.releaseVersion.Length > 100 || manifest.releaseVersion.Any(Char.IsControl)) throw new SetupException("A releaseVersion is required.");
            if (!String.Equals(manifest.architecture, "arm64", StringComparison.OrdinalIgnoreCase)) throw new SetupException("The release architecture must be arm64.");
            if (manifest.payload == null || manifest.payload.size <= 0 || manifest.payload.unpackedBytes <= 0) throw new SetupException("Payload size and unpackedBytes must be positive exact byte counts.");
            Safety.ValidateHash(manifest.payload.sha256, "Payload sha256");
            manifest.payload.entryPoint = Safety.RelativePath(manifest.payload.entryPoint);
            string extension = Path.GetExtension(manifest.payload.entryPoint).ToLowerInvariant();
            if (extension != ".ps1" && extension != ".exe" && extension != ".cmd" && extension != ".bat") throw new SetupException("The setup entryPoint must be a relative .ps1, .exe, .cmd, or .bat path.");
            if (manifest.models == null) manifest.models = new List<ModelSpec>();
            if (manifest.models.Count > 1000) throw new SetupException("The model inventory exceeds the supported limit.");
            var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (ModelSpec model in manifest.models)
            {
                if (model == null || model.size <= 0) throw new SetupException("Every model needs a positive exact byte count.");
                model.path = Safety.RelativePath(model.path);
                if (model.path.Equals(".livinghome-install.json", StringComparison.OrdinalIgnoreCase) || model.path.Equals(".setup-owner", StringComparison.OrdinalIgnoreCase) || model.path.Equals(".livinghome-setup.log", StringComparison.OrdinalIgnoreCase)) throw new SetupException("A model uses an installer-reserved filename.");
                if (!paths.Add(model.path)) throw new SetupException("The model inventory contains duplicate paths.");
                Safety.ValidateHash(model.sha256, "Model sha256");
            }
            if (manifest.minimumFreeBytes < 0) throw new SetupException("minimumFreeBytes cannot be negative.");
            try { checked { long total = manifest.payload.size + manifest.payload.unpackedBytes + manifest.models.Sum(x => x.size); if (total <= 0) throw new OverflowException(); } }
            catch (OverflowException) { throw new SetupException("The release sizes exceed the supported range."); }
        }
    }

    public static class Download
    {
        private const int BufferSize = 1024 * 1024;
        public static string Hex(byte[] bytes) { return BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant(); }
        public static string HashFile(string path, CancellationToken cancel)
        {
            Safety.NoReparse(path);
            using (var sha = SHA256.Create())
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize, FileOptions.SequentialScan))
            {
                byte[] buffer = new byte[BufferSize];
                int read;
                while ((read = stream.Read(buffer, 0, buffer.Length)) > 0)
                {
                    cancel.ThrowIfCancellationRequested();
                    sha.TransformBlock(buffer, 0, read, buffer, 0);
                }
                sha.TransformFinalBlock(new byte[0], 0, 0);
                return Hex(sha.Hash);
            }
        }
        public static void Verify(string path, long size, string sha, CancellationToken cancel)
        {
            Safety.NoReparse(path);
            if (new FileInfo(path).Length != size) throw new SetupException("The downloaded file has the wrong size. Expected " + size + " bytes.");
            if (!HashFile(path, cancel).Equals(sha, StringComparison.OrdinalIgnoreCase)) throw new SetupException("SHA256 verification failed. The file is corrupt or does not match this release.");
        }
        public static HttpWebResponse OpenResponse(Uri uri, long offset, CancellationToken cancel)
        {
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            for (int hop = 0; hop <= 5; hop++)
            {
                bool permitted = uri.Scheme == "https";
#if INSTALLER_TESTS
                permitted = permitted || uri.Scheme == "http" && uri.IsLoopback;
#endif
                if (!permitted || !String.IsNullOrEmpty(uri.UserInfo)) throw new SetupException("Downloads and redirects must use HTTPS without URL credentials.");
                var request = (HttpWebRequest)WebRequest.Create(uri);
                request.AllowAutoRedirect = false;
                request.AutomaticDecompression = DecompressionMethods.None;
                request.Timeout = 30000;
                request.ReadWriteTimeout = 30000;
                request.UserAgent = "LivingHomeSetup/1.0";
                if (offset > 0) request.AddRange(offset);
                HttpWebResponse response;
                using (cancel.Register(request.Abort))
                {
                    try { response = (HttpWebResponse)request.GetResponse(); }
                    catch (WebException) { cancel.ThrowIfCancellationRequested(); throw; }
                }
                int code = (int)response.StatusCode;
                if (code == 301 || code == 302 || code == 303 || code == 307 || code == 308)
                {
                    string location = response.Headers[HttpResponseHeader.Location];
                    response.Dispose();
                    Uri next;
                    if (hop == 5 || String.IsNullOrEmpty(location) || !Uri.TryCreate(uri, location, out next)) throw new SetupException("The download redirect is invalid or has too many hops.");
                    uri = next;
                    continue;
                }
                return response;
            }
            throw new SetupException("The download redirect limit was exceeded.");
        }
        public static string Verified(Source source, string cacheDirectory, long size, string sha, string label, CancellationToken cancel, Action<ProgressUpdate> progress)
        {
            Safety.ValidateHash(sha, label + " SHA256");
            if (size <= 0) throw new SetupException("The expected download size must be positive.");
            Safety.NoReparse(cacheDirectory);
            Directory.CreateDirectory(cacheDirectory);
            Safety.NoReparse(cacheDirectory);
            string final = Path.Combine(cacheDirectory, sha.ToLowerInvariant() + ".verified");
            string partial = Path.Combine(cacheDirectory, sha.ToLowerInvariant() + ".partial");
            Safety.NoReparse(final);
            Safety.NoReparse(partial);
            if (File.Exists(final))
            {
                progress(new ProgressUpdate { Message = "Verifying cached " + label, Total = size });
                try { Verify(final, size, sha, cancel); return final; }
                catch (SetupException) { File.Delete(final); }
            }
            if (File.Exists(partial) && new FileInfo(partial).Length > size) File.Delete(partial);
            for (int attempt = 1; attempt <= 3; attempt++)
            {
                cancel.ThrowIfCancellationRequested();
                Safety.NoReparse(partial);
                long offset = File.Exists(partial) ? new FileInfo(partial).Length : 0;
                if (offset < size)
                {
                    try
                    {
                        if (source.IsLocal)
                        {
                            Safety.NoReparse(source.FilePath);
                            if (!File.Exists(source.FilePath) || new FileInfo(source.FilePath).Length != size) throw new SetupException("Local " + label + " is missing or has the wrong size: " + source.FilePath);
                            using (var input = new FileStream(source.FilePath, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize, FileOptions.SequentialScan))
                            {
                                input.Position = offset;
                                CopyPartial(input, partial, offset, size, label, cancel, progress);
                            }
                        }
                        else
                        {
                            using (HttpWebResponse response = OpenResponse(source.RemoteUri, offset, cancel))
                            {
                                int status = (int)response.StatusCode;
                                if (offset > 0 && status == 200) offset = 0;
                                else if (status == 206)
                                {
                                    Match range = Regex.Match(response.Headers[HttpResponseHeader.ContentRange] ?? "", "^bytes ([0-9]+)-([0-9]+)/([0-9]+)$");
                                    long start, end, total;
                                    if (!range.Success || !Int64.TryParse(range.Groups[1].Value, out start) || !Int64.TryParse(range.Groups[2].Value, out end) || !Int64.TryParse(range.Groups[3].Value, out total) || start != offset || total != size || end < start || end >= size)
                                        throw new SetupException("The server returned an invalid resume range. The partial file was preserved.");
                                    if (end != size - 1) throw new SetupException("The server returned an incomplete byte range. Use a server that supports complete range downloads.");
                                }
                                else if (status != 200) throw new SetupException("The download server returned HTTP " + status + ".");
                                if (response.ContentLength >= 0 && response.ContentLength != size - offset) throw new SetupException("The download server's byte count does not match the release manifest.");
                                if (!String.IsNullOrEmpty(response.Headers[HttpResponseHeader.ContentEncoding]) && response.Headers[HttpResponseHeader.ContentEncoding] != "identity") throw new SetupException("Compressed HTTP transfer encoding is unsupported for exact-byte downloads.");
                                using (Stream input = response.GetResponseStream()) CopyPartial(input, partial, offset, size, label, cancel, progress);
                            }
                        }
                    }
                    catch (WebException)
                    {
                        cancel.ThrowIfCancellationRequested();
                        if (attempt == 3) throw new SetupException("Network download failed after three attempts. The partial " + label + " is saved; retry to resume. Check the HTTPS origin, network, and proxy settings.");
                        progress(new ProgressUpdate { Message = "Connection interrupted. Resuming " + label + " (attempt " + (attempt + 1) + ")", Current = File.Exists(partial) ? new FileInfo(partial).Length : 0, Total = size });
                        if (cancel.WaitHandle.WaitOne(500 * attempt)) cancel.ThrowIfCancellationRequested();
                        continue;
                    }
                    catch (IOException)
                    {
                        cancel.ThrowIfCancellationRequested();
                        if (attempt == 3 || source.IsLocal) throw;
                        if (cancel.WaitHandle.WaitOne(500 * attempt)) cancel.ThrowIfCancellationRequested();
                        continue;
                    }
                }
                progress(new ProgressUpdate { Message = "Checking SHA256 for " + label, Current = size, Total = size });
                try { Verify(partial, size, sha, cancel); }
                catch (SetupException) { Safety.NoReparse(partial); File.Delete(partial); throw; }
                Safety.NoReparse(final);
                File.Move(partial, final);
                return final;
            }
            throw new SetupException("The download could not be completed.");
        }
        private static void CopyPartial(Stream input, string partial, long offset, long size, string label, CancellationToken cancel, Action<ProgressUpdate> progress)
        {
            Safety.NoReparse(partial);
            using (var output = new FileStream(partial, offset == 0 ? FileMode.Create : FileMode.Open, FileAccess.Write, FileShare.None, BufferSize))
            using (cancel.Register(input.Dispose))
            {
                if (output.Length != offset) throw new SetupException("The partial download changed during resume.");
                output.Position = offset;
                byte[] buffer = new byte[BufferSize];
                long current = offset;
                var clock = Stopwatch.StartNew();
                int read;
                while ((read = ReadWithCancellation(input, buffer, (int)Math.Min(buffer.Length, size - current + 1), cancel)) > 0)
                {
                    cancel.ThrowIfCancellationRequested();
                    if (current + read > size) throw new SetupException("The source sent more bytes than declared in the release manifest.");
                    output.Write(buffer, 0, read);
                    current += read;
                    if (clock.ElapsedMilliseconds >= 150 || current == size || current - offset == read)
                    {
                        progress(new ProgressUpdate { Message = "Downloading " + label, Current = current, Total = size });
                        clock.Restart();
                    }
                }
                output.Flush(true);
                if (current != size) throw new IOException("The download ended early. The partial file is saved for retry.");
            }
        }
        private static int ReadWithCancellation(Stream stream, byte[] buffer, int count, CancellationToken token)
        {
            token.ThrowIfCancellationRequested();
            try { return stream.Read(buffer, 0, count); }
            catch (Exception) { token.ThrowIfCancellationRequested(); throw; }
        }
    }

    public static class Archive
    {
        public static void ExtractVerifiedZip(string zipPath, string staging, long expectedBytes, string entryPoint, CancellationToken cancel, Action<ProgressUpdate> progress)
        {
            Safety.NoReparse(zipPath);
            Safety.NoReparse(staging);
            using (var input = new FileStream(zipPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var zip = new ZipArchive(input, ZipArchiveMode.Read))
            {
                if (zip.Entries.Count > 200000) throw new SetupException("The ZIP has too many entries.");
                var entries = new List<Tuple<ZipArchiveEntry, string, bool>>();
                var paths = new Dictionary<string, bool>(StringComparer.OrdinalIgnoreCase);
                long total = 0;
                foreach (ZipArchiveEntry entry in zip.Entries)
                {
                    cancel.ThrowIfCancellationRequested();
                    bool directory = entry.FullName.EndsWith("/") || entry.FullName.EndsWith("\\");
                    string relative = Safety.RelativePath(directory ? entry.FullName.TrimEnd('/', '\\') : entry.FullName);
                    if (relative.Equals(".setup-owner", StringComparison.OrdinalIgnoreCase) || relative.Equals(".livinghome-install.json", StringComparison.OrdinalIgnoreCase) || relative.Equals(".livinghome-setup.log", StringComparison.OrdinalIgnoreCase)) throw new SetupException("The ZIP uses an installer-reserved filename.");
                    int unixType = (entry.ExternalAttributes >> 16) & 0xF000;
                    if (unixType == 0xA000 || (entry.ExternalAttributes & 0x400) != 0) throw new SetupException("ZIP links and reparse points are unsupported: " + relative);
                    if (paths.ContainsKey(relative)) throw new SetupException("The ZIP contains duplicate Windows paths: " + relative);
                    paths.Add(relative, directory);
                    if (directory && entry.Length != 0) throw new SetupException("A ZIP directory unexpectedly contains data.");
                    try { checked { total += entry.Length; } } catch (OverflowException) { throw new SetupException("The ZIP size exceeds the supported range."); }
                    if (total > expectedBytes) throw new SetupException("The ZIP uncompressed size exceeds payload.unpackedBytes.");
                    entries.Add(Tuple.Create(entry, relative, directory));
                }
                if (total != expectedBytes) throw new SetupException("The ZIP uncompressed size does not match payload.unpackedBytes.");
                foreach (var item in entries)
                {
                    string parent = Path.GetDirectoryName(item.Item2);
                    while (!String.IsNullOrEmpty(parent))
                    {
                        bool isDirectory;
                        if (paths.TryGetValue(parent, out isDirectory) && !isDirectory) throw new SetupException("A ZIP file is also used as a directory: " + parent);
                        parent = Path.GetDirectoryName(parent);
                    }
                }
                bool setupIsDirectory;
                if (!paths.TryGetValue(Safety.RelativePath(entryPoint), out setupIsDirectory) || setupIsDirectory) throw new SetupException("The payload does not contain its declared setup entryPoint.");
                long complete = 0;
                byte[] buffer = new byte[1024 * 1024];
                foreach (var item in entries)
                {
                    cancel.ThrowIfCancellationRequested();
                    string target = Safety.Under(staging, item.Item2);
                    if (target.Length > 259) throw new SetupException("An extracted path exceeds the Windows path limit. Choose a shorter installation directory.");
                    if (item.Item3) { Safety.CreateDirectories(staging, target); continue; }
                    Safety.CreateDirectories(staging, Path.GetDirectoryName(target));
                    Safety.NoReparse(target);
                    using (Stream content = item.Item1.Open())
                    using (var output = new FileStream(target, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                    {
                        long written = 0;
                        int count;
                        while ((count = content.Read(buffer, 0, buffer.Length)) > 0)
                        {
                            cancel.ThrowIfCancellationRequested();
                            if (written + count > item.Item1.Length) throw new SetupException("A ZIP entry exceeds its declared size.");
                            output.Write(buffer, 0, count);
                            written += count;
                            complete += count;
                            progress(new ProgressUpdate { Message = "Extracting payload", Current = complete, Total = total });
                        }
                        if (written != item.Item1.Length) throw new SetupException("A ZIP entry is truncated.");
                    }
                }
            }
        }
    }

    public static class Readiness
    {
        private const long GiB = 1024L * 1024 * 1024;
        [StructLayout(LayoutKind.Sequential)] private struct SystemInfo
        {
            public ushort architecture, reserved; public uint pageSize; public IntPtr minimumAddress, maximumAddress, activeMask;
            public uint processors, processorType, allocationGranularity; public ushort processorLevel, processorRevision;
        }
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)] private class MemoryStatus
        {
            public uint length = (uint)Marshal.SizeOf(typeof(MemoryStatus)); public uint load;
            public ulong physicalTotal, physicalAvailable, pageFileTotal, pageFileAvailable, virtualTotal, virtualAvailable, extendedAvailable;
        }
        [DllImport("kernel32.dll")] private static extern void GetNativeSystemInfo(out SystemInfo info);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool IsWow64Process2(IntPtr process, out ushort processMachine, out ushort nativeMachine);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool GlobalMemoryStatusEx([In, Out] MemoryStatus status);
        public static string FormatBytes(long value) { return (value / (double)GiB).ToString("0.0") + " GiB"; }
        public static ReadinessReport Check(LoadedRelease release, string target, string cache)
        {
            var report = new ReadinessReport { checkedAtUtc = DateTime.UtcNow.ToString("o"), releaseVersion = release.Manifest.releaseVersion, manifestSha256 = release.ManifestHash, manifestPinned = release.IsPinned, installDirectory = target, checks = new List<CheckResult>() };
            Action<string, string, string> add = (name, status, detail) => report.checks.Add(new CheckResult { name = name, status = status, detail = detail });
            try { target = Safety.NewDestination(target); report.installDirectory = target; add("Installation directory", "pass", "A new directory under an existing local parent."); }
            catch (Exception ex) { add("Installation directory", "fail", ex.Message); }
            SystemInfo info; GetNativeSystemInfo(out info);
            bool arm64 = info.architecture == 12;
            string architectureDetail = "processor architecture code " + info.architecture;
            try
            {
                ushort processMachine, nativeMachine;
                if (IsWow64Process2(new IntPtr(-1), out processMachine, out nativeMachine))
                {
                    arm64 = nativeMachine == 0xAA64;
                    architectureDetail = "native machine 0x" + nativeMachine.ToString("X4") + ", process machine 0x" + processMachine.ToString("X4");
                }
            }
            catch (EntryPointNotFoundException) { }
            add("Native architecture", arm64 ? "pass" : "fail", arm64 ? "Windows ARM64 (" + architectureDetail + ")" : "This release requires native Windows ARM64. Detected " + architectureDetail + ".");
            try
            {
                using (var search = new ManagementObjectSearcher("SELECT Caption, BuildNumber FROM Win32_OperatingSystem"))
                {
                    bool found = false;
                    foreach (ManagementObject os in search.Get())
                    {
                        int build; found = Int32.TryParse(Convert.ToString(os["BuildNumber"]), out build);
                        add("Windows version", found && build >= 22000 ? "pass" : "fail", Convert.ToString(os["Caption"]) + " (build " + Convert.ToString(os["BuildNumber"]) + "). Windows 11 or newer is required.");
                    }
                    if (!found) add("Windows version", "fail", "Windows build could not be verified.");
                }
            }
            catch (Exception ex) { add("Windows version", "fail", "Cannot query Windows readiness: " + ex.Message); }
            try
            {
                int value = 0;
                using (RegistryKey key = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full")) if (key != null) value = Convert.ToInt32(key.GetValue("Release", 0));
                add(".NET Framework", value >= 533320 ? "pass" : "fail", value >= 533320 ? ".NET Framework 4.8.1 or newer is installed." : "Install Microsoft .NET Framework 4.8.1 for native Windows ARM64 support. Detected release value " + value + ".");
            }
            catch (Exception ex) { add(".NET Framework", "fail", "Cannot inspect .NET Framework: " + ex.Message); }
            try
            {
                var names = new List<string>();
                using (var search = new ManagementObjectSearcher("SELECT Name, DriverVersion FROM Win32_VideoController"))
                    foreach (ManagementObject gpu in search.Get())
                    {
                        string name = Convert.ToString(gpu["Name"]);
                        if (name.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) >= 0) names.Add(name + " (driver " + Convert.ToString(gpu["DriverVersion"]) + ")");
                    }
                add("NVIDIA GPU", names.Count > 0 ? "pass" : "fail", names.Count > 0 ? String.Join("; ", names) : "No NVIDIA GPU was reported. Install the device's Windows ARM64 NVIDIA driver and retry.");
                add("CUDA runtime", "warning", "GPU presence does not prove CUDA/model compatibility. Verify the release's ARM64 runtime on the target device.");
            }
            catch (Exception ex) { add("NVIDIA GPU", "fail", "Cannot query the GPU: " + ex.Message); }
            var memory = new MemoryStatus();
            if (GlobalMemoryStatusEx(memory))
            {
                long largest = release.Manifest.models.Count == 0 ? 0 : release.Manifest.models.Max(x => x.size);
                ulong needed = (ulong)largest + 8UL * (ulong)GiB;
                add("Available commit", memory.pageFileAvailable >= needed ? "pass" : "fail", FormatBytes((long)memory.pageFileAvailable) + " available; requires at least " + FormatBytes((long)needed) + " for the largest declared model plus 8 GiB. Runtime peak use still needs target validation.");
                add("Physical memory", memory.physicalTotal >= 120UL * (ulong)GiB ? "pass" : "warning", FormatBytes((long)memory.physicalTotal) + " visible to Windows. GPU reservation can reduce this value; the intended target has 128 GB shared memory. Setup does not alter reservation.");
            }
            else add("Available commit", "fail", "Windows memory/commit information could not be read.");
            try
            {
                Safety.NoReparse(cache);
                long models = release.Manifest.models.Sum(x => x.size);
                long installNeed, cacheNeed;
                checked { installNeed = Math.Max(release.Manifest.minimumFreeBytes, release.Manifest.payload.unpackedBytes + models + 4 * GiB); cacheNeed = release.Manifest.payload.size + models + GiB; }
                string targetDrive = Path.GetPathRoot(Path.GetFullPath(target));
                string cacheDrive = Path.GetPathRoot(Path.GetFullPath(cache));
                long available = new DriveInfo(targetDrive).AvailableFreeSpace;
                if (targetDrive.Equals(cacheDrive, StringComparison.OrdinalIgnoreCase))
                {
                    long required = checked(installNeed + cacheNeed);
                    add("Storage", available >= required ? "pass" : "fail", FormatBytes(available) + " available on " + targetDrive + "; requires " + FormatBytes(required) + " for payload, models, extraction and safety reserve.");
                }
                else
                {
                    add("Installation storage", available >= installNeed ? "pass" : "fail", FormatBytes(available) + " available; requires " + FormatBytes(installNeed) + ".");
                    long cacheAvailable = new DriveInfo(cacheDrive).AvailableFreeSpace;
                    add("Download cache storage", cacheAvailable >= cacheNeed ? "pass" : "fail", FormatBytes(cacheAvailable) + " available; requires " + FormatBytes(cacheNeed) + " on " + cacheDrive + ".");
                }
            }
            catch (Exception ex) { add("Storage", "fail", "Cannot validate installation/cache storage: " + ex.Message); }
            add("Publisher trust", release.IsPinned ? "pass" : "warning", release.IsPinned ? "The supplied SHA256 pin matches the exact manifest bytes." : "No manifest pin supplied. Confirm the displayed release publisher before installation; unattended installation requires a pin.");
            report.canInstall = report.checks.All(x => x.status != "fail");
            return report;
        }
    }

    public sealed class InstallResult
    {
        public string InstallDirectory;
        public int SetupExitCode;
    }
    public static class InstallerEngine
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern bool CreateDirectory(string name, IntPtr attributes);
        public static string CacheRoot
        {
            get { return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LivingHome", "Installer", "Cache"); }
        }
        public static string ReleaseCache(LoadedRelease release) { return Path.Combine(CacheRoot, release.ManifestHash); }
        public static InstallResult Install(LoadedRelease release, string target, CancellationToken cancel, Action<ProgressUpdate> progress)
        {
            target = Safety.NewDestination(target);
            string cache = ReleaseCache(release);
            ReadinessReport readiness = Readiness.Check(release, target, cache);
            if (!readiness.canInstall) throw new SetupException("Readiness checks failed. Resolve the reported problems before installation.");
            string payload = Download.Verified(release.PayloadSource, cache, release.Manifest.payload.size, release.Manifest.payload.sha256, "release payload", cancel, progress);
            var models = new List<string>();
            for (int i = 0; i < release.Manifest.models.Count; i++)
            {
                ModelSpec model = release.Manifest.models[i];
                models.Add(Download.Verified(release.ModelSources[i], cache, model.size, model.sha256, "model " + (i + 1) + " of " + release.Manifest.models.Count, cancel, progress));
            }
            cancel.ThrowIfCancellationRequested();
            target = Safety.NewDestination(target);
            string owner = Guid.NewGuid().ToString("N");
            string staging = target + ".setup-" + owner.Substring(0, 8);
            Safety.NoReparse(staging);
            if (!CreateDirectory(staging, IntPtr.Zero)) throw new SetupException("Cannot create a new staging directory. Windows error " + Marshal.GetLastWin32Error() + ".");
            File.WriteAllText(Path.Combine(staging, ".setup-owner"), owner, new UTF8Encoding(false));
            bool committed = false;
            try
            {
                Download.Verify(payload, release.Manifest.payload.size, release.Manifest.payload.sha256, cancel);
                Archive.ExtractVerifiedZip(payload, staging, release.Manifest.payload.unpackedBytes, release.Manifest.payload.entryPoint, cancel, progress);
                for (int i = 0; i < models.Count; i++)
                {
                    cancel.ThrowIfCancellationRequested();
                    ModelSpec model = release.Manifest.models[i];
                    string destination = Safety.Under(staging, model.path);
                    Safety.CreateDirectories(staging, Path.GetDirectoryName(destination));
                    Safety.NoReparse(destination);
                    progress(new ProgressUpdate { Message = "Placing model " + (i + 1) + " of " + models.Count, Total = model.size });
                    CopyVerifiedModel(models[i], destination, model.size, model.sha256, cancel, progress);
                }
                Safety.NoReparse(staging);
                Safety.NewDestination(target);
                cancel.ThrowIfCancellationRequested();
                WriteReceipt(staging, release, "payload-ready", null);
                Directory.Move(staging, target);
                committed = true;
                File.Delete(Path.Combine(target, ".setup-owner"));
                progress(new ProgressUpdate { Message = "Launching the release setup. Follow any release-specific prompts." });
                // Setup is intentionally not cancellable: terminating arbitrary release setup can leave the installation damaged.
                int exit = LaunchSetup(target, release.Manifest.payload.entryPoint);
                WriteReceipt(target, release, exit == 0 ? "setup-complete" : "setup-failed", exit);
                if (exit != 0) throw new SetupException("Release setup exited with code " + exit + ". The extracted installation is preserved at " + target + ". Inspect " + Path.Combine(target, ".livinghome-setup.log") + "; choose a new directory for a fresh install.");
                return new InstallResult { InstallDirectory = target, SetupExitCode = exit };
            }
            catch (Exception ex)
            {
                if (!committed && Directory.Exists(staging))
                {
                    try { Safety.DeleteOwnedStaging(staging, owner); }
                    catch (Exception cleanup) { throw new SetupException(ex.Message + " Staging was preserved at " + staging + ": " + cleanup.Message); }
                }
                if (committed && !(ex is SetupException)) throw new SetupException("Release setup could not complete: " + ex.Message + " The installation is preserved at " + target + ".");
                throw;
            }
        }
        private static void CopyVerifiedModel(string source, string destination, long size, string expectedHash, CancellationToken cancel, Action<ProgressUpdate> progress)
        {
            Safety.NoReparse(source);
            using (var input = new FileStream(source, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan))
            using (var output = new FileStream(destination, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            using (var sha = SHA256.Create())
            {
                byte[] buffer = new byte[1024 * 1024]; long total = 0; int count; var clock = Stopwatch.StartNew();
                while ((count = input.Read(buffer, 0, buffer.Length)) > 0)
                {
                    cancel.ThrowIfCancellationRequested(); output.Write(buffer, 0, count); sha.TransformBlock(buffer, 0, count, buffer, 0); total += count;
                    if (clock.ElapsedMilliseconds >= 150 || total == size) { progress(new ProgressUpdate { Message = "Placing verified model", Current = total, Total = size }); clock.Restart(); }
                }
                output.Flush(true);
                sha.TransformFinalBlock(new byte[0], 0, 0);
                if (total != size) throw new SetupException("The verified model cache changed during installation.");
                if (!Download.Hex(sha.Hash).Equals(expectedHash, StringComparison.OrdinalIgnoreCase)) throw new SetupException("The model SHA256 changed while copying from cache. The installation was stopped.");
            }
        }
        private static int LaunchSetup(string directory, string relative)
        {
            string entry = Safety.Under(directory, relative);
            Safety.NoReparse(entry);
            if (!File.Exists(entry)) throw new SetupException("The declared setup entryPoint is missing.");
            string extension = Path.GetExtension(entry).ToLowerInvariant();
            ProcessStartInfo info;
            if (extension == ".ps1")
            {
                string powershell = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), @"System32\WindowsPowerShell\v1.0\powershell.exe");
                info = new ProcessStartInfo(powershell, "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"" + entry + "\"");
            }
            else if (extension == ".cmd" || extension == ".bat")
            {
                if (entry.IndexOf('%') >= 0 || entry.IndexOf('!') >= 0) throw new SetupException("Batch entry point paths cannot contain percent or exclamation characters.");
                // The /s /c convention requires an outer quoted string around the quoted batch filename.
                info = new ProcessStartInfo(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), @"System32\cmd.exe"), "/d /s /c \"\"" + entry + "\"\"");
            }
            else if (extension == ".exe") info = new ProcessStartInfo(entry);
            else throw new SetupException("The setup entryPoint extension is unsupported.");
            info.WorkingDirectory = directory;
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            info.RedirectStandardInput = true;
            info.RedirectStandardOutput = true;
            info.RedirectStandardError = true;
            string logPath = Path.Combine(directory, ".livinghome-setup.log");
            Safety.NoReparse(logPath);
            using (var logFile = new FileStream(logPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
            using (var writer = new StreamWriter(logFile, new UTF8Encoding(false)))
            using (var process = new Process { StartInfo = info })
            {
                object gate = new object();
                DataReceivedEventHandler stdout = (sender, line) => { if (line.Data != null) lock (gate) { writer.WriteLine(line.Data); writer.Flush(); } };
                DataReceivedEventHandler stderr = (sender, line) => { if (line.Data != null) lock (gate) { writer.WriteLine("ERROR: " + line.Data); writer.Flush(); } };
                process.OutputDataReceived += stdout; process.ErrorDataReceived += stderr;
                process.Start(); process.StandardInput.Close(); process.BeginOutputReadLine(); process.BeginErrorReadLine();
                process.WaitForExit();
                return process.ExitCode;
            }
        }
        private static void WriteReceipt(string directory, LoadedRelease release, string status, int? exit)
        {
            string receipt = Path.Combine(directory, ".livinghome-install.json");
            Safety.NoReparse(receipt);
            var data = new Dictionary<string, object> { { "installerVersion", "1.0.0" }, { "releaseVersion", release.Manifest.releaseVersion }, { "manifestSha256", release.ManifestHash }, { "architecture", release.Manifest.architecture }, { "status", status }, { "updatedAtUtc", DateTime.UtcNow.ToString("o") }, { "setupExitCode", exit } };
            File.WriteAllText(receipt, new JavaScriptSerializer().Serialize(data), new UTF8Encoding(false));
        }
    }
}
