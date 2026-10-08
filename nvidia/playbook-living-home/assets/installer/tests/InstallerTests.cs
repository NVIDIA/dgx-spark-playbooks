using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using Microsoft.Win32.SafeHandles;

namespace LivingHome.Setup
{
    public static class InstallerTests
    {
        private static string root, bootstrapper;
        private static readonly List<object> results = new List<object>();
        private static readonly List<string> junctions = new List<string>();
        private static int failures;
        private static readonly Action<ProgressUpdate> Quiet = update => { };
        public static int Main(string[] args)
        {
            string report = null;
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--report" && i + 1 < args.Length) report = args[++i];
                else if (args[i] == "--bootstrapper" && i + 1 < args.Length) bootstrapper = args[++i];
                else throw new Exception("Unknown test argument.");
            }
            root = Path.Combine(Path.GetTempPath(), "LivingHomeSetupTests-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            Console.WriteLine("Synthetic safety fixtures: " + root);
            try
            {
                Run("Manifest: pin, schema, HTTPS, paths and exact sizes", ManifestValidation);
                Run("ZIP: valid payload extracts without launching code", ValidArchive);
                Run("ZIP: traversal, rooted paths, ADS, device names and duplicate aliases rejected before extraction", ArchiveTraversal);
                Run("ZIP: symbolic links, file/directory conflicts, missing entry point and size mismatch rejected", ArchiveMetadata);
                Run("Windows junctions: extraction, cache and destination cannot escape through reparse points", ReparseGuard);
                Run("Existing install and extracted files are never overwritten", NoOverwrite);
                Run("Local download: exact size/hash required; corrupt cache is rejected and repaired", DigestChecks);
                Run("Cancelled partial local download resumes and verifies", CancellationResume);
                Run("HTTP transfer interrupted mid-body resumes using validated Range", InterruptedHttp);
                Run("HTTP server ignoring Range restarts safely", IgnoredRange);
                Run("Invalid Content-Range is rejected without touching partial bytes", InvalidRange);
                Run("HTTP oversized/wrong byte counts are rejected", WrongByteCount);
                Run("Production EXE check-only and unattended pin gate do not install or download assets", BinaryCheckOnly);
            }
            finally
            {
                foreach (string junction in junctions)
                    if (Directory.Exists(junction)) Directory.Delete(junction, false);
                Safety.NoReparse(root);
                Directory.Delete(root, true);
            }
            if (report != null)
            {
                Safety.NoReparse(report);
                File.WriteAllText(report, new JavaScriptSerializer().Serialize(new { testSuite = "LivingHomeSetup synthetic Windows safety tests", completedAtUtc = DateTime.UtcNow.ToString("o"), passed = results.Count - failures, failed = failures, fixturesRoot = root, fixturesRemoved = !Directory.Exists(root), results = results }), new UTF8Encoding(false));
            }
            Console.WriteLine((results.Count - failures) + " passed; " + failures + " failed. No release entry point or runtime service was launched.");
            return failures == 0 ? 0 : 1;
        }
        private static void Run(string name, Action test)
        {
            var timer = Stopwatch.StartNew();
            try { test(); Console.WriteLine("PASS " + name); results.Add(new { name = name, status = "pass", milliseconds = timer.ElapsedMilliseconds }); }
            catch (Exception ex) { failures++; Console.WriteLine("FAIL " + name + ": " + ex); results.Add(new { name = name, status = "fail", detail = ex.ToString(), milliseconds = timer.ElapsedMilliseconds }); }
        }
        private static void Assert(bool condition, string message) { if (!condition) throw new Exception(message); }
        private static void Reject(Action action, string message)
        {
            bool rejected = false;
            try { action(); } catch (SetupException) { rejected = true; } catch (IOException) { rejected = true; }
            Assert(rejected, message);
        }
        private static string Case(string label) { string path = Path.Combine(root, label + "-" + Guid.NewGuid().ToString("N").Substring(0, 6)); Directory.CreateDirectory(path); return path; }
        private static string Hash(byte[] bytes) { using (var sha = SHA256.Create()) return Download.Hex(sha.ComputeHash(bytes)); }
        private static byte[] Data(int length) { byte[] bytes = new byte[length]; new Random(7124).NextBytes(bytes); return bytes; }
        private static string Zip(string directory, params string[] names)
        {
            string path = Path.Combine(directory, "fixture-" + Guid.NewGuid().ToString("N") + ".zip");
            using (var stream = new FileStream(path, FileMode.CreateNew))
            using (var zip = new ZipArchive(stream, ZipArchiveMode.Create))
                foreach (string name in names)
                {
                    ZipArchiveEntry entry = zip.CreateEntry(name);
                    if (!name.EndsWith("/")) using (Stream data = entry.Open()) data.WriteByte(42);
                }
            return path;
        }
        private static ReleaseManifest Fixture(string directory)
        {
            string zip = Zip(directory, "setup.ps1", "app/file.txt");
            return new ReleaseManifest { schemaVersion = 1, releaseVersion = "TEST-DO-NOT-INSTALL-" + Guid.NewGuid().ToString("N"), architecture = "arm64", payload = new PayloadSpec { url = zip, size = new FileInfo(zip).Length, sha256 = Download.HashFile(zip, CancellationToken.None), unpackedBytes = 2, entryPoint = "setup.ps1" }, models = new List<ModelSpec>() };
        }
        private static void ManifestValidation()
        {
            string directory = Case("manifest");
            ReleaseManifest model = Fixture(directory);
            string manifest = Path.Combine(directory, "release.json");
            File.WriteAllText(manifest, new JavaScriptSerializer().Serialize(model), new UTF8Encoding(false));
            string hash = Download.HashFile(manifest, CancellationToken.None);
            LoadedRelease loaded = ManifestReader.Read(manifest, hash, CancellationToken.None);
            Assert(loaded.IsPinned && loaded.ManifestHash == hash && loaded.PayloadSource.IsLocal, "Valid pinned local manifest must load.");
            Reject(() => ManifestReader.Read(manifest, new string('0', 64), CancellationToken.None), "Wrong manifest pin accepted.");
            Reject(() => ManifestReader.ResolveManifest("http://example.invalid/release.json"), "Plain HTTP manifest accepted.");
            Reject(() => ManifestReader.ResolveAsset(new Source { RemoteUri = new Uri("https://example.invalid/release.json") }, "file:///C:/Windows/not-a-release.zip"), "Remote manifest pointed to local code.");
            Reject(() => ManifestReader.ResolveAsset(loaded.ManifestSource, "http://example.invalid/payload.zip"), "HTTP payload accepted.");
            model.schemaVersion = 2; Reject(() => ManifestReader.Validate(model), "Wrong schema accepted."); model.schemaVersion = 1;
            model.payload.entryPoint = "../outside.ps1"; Reject(() => ManifestReader.Validate(model), "Traversal entrypoint accepted."); model.payload.entryPoint = "setup.ps1";
            model.models.Add(new ModelSpec { path = "models/CON.gguf", url = "local.gguf", size = 1, sha256 = hash });
            Reject(() => ManifestReader.Validate(model), "Reserved device model path accepted."); model.models.Clear();
            model.payload.size = 0; Reject(() => ManifestReader.Validate(model), "Missing exact payload size accepted.");
        }
        private static void ValidArchive()
        {
            string directory = Case("valid-zip"), stage = Path.Combine(directory, "stage"); Directory.CreateDirectory(stage);
            string zip = Zip(directory, "setup.ps1", "assets/", "assets/test.txt");
            Archive.ExtractVerifiedZip(zip, stage, 2, "setup.ps1", CancellationToken.None, Quiet);
            Assert(File.ReadAllBytes(Path.Combine(stage, "assets", "test.txt"))[0] == 42, "Valid fixture was not extracted.");
        }
        private static void ArchiveTraversal()
        {
            foreach (string unsafeName in new[] { "../escape.txt", "..\\escape.txt", "/absolute.txt", "C:/absolute.txt", "folder/file.txt:stream", "NUL.txt", "folder/trailing. ", "folder/../alias.txt", "\\server\\file.txt", "folder//file.txt" })
            {
                string directory = Case("unsafe-zip"), stage = Path.Combine(directory, "stage"); Directory.CreateDirectory(stage);
                string zip = Zip(directory, "setup.ps1", unsafeName);
                Reject(() => Archive.ExtractVerifiedZip(zip, stage, 2, "setup.ps1", CancellationToken.None, Quiet), "Unsafe ZIP name accepted: " + unsafeName);
                Assert(Directory.GetFileSystemEntries(stage).Length == 0, "ZIP preflight wrote data before rejecting unsafe path.");
            }
            string duplicate = Case("duplicate"), output = Path.Combine(duplicate, "stage"); Directory.CreateDirectory(output);
            Reject(() => Archive.ExtractVerifiedZip(Zip(duplicate, "setup.ps1", "Case.txt", "case.txt"), output, 3, "setup.ps1", CancellationToken.None, Quiet), "Case-insensitive duplicate paths accepted.");
        }
        private static void ArchiveMetadata()
        {
            string directory = Case("metadata"), stage = Path.Combine(directory, "stage"); Directory.CreateDirectory(stage);
            string zipPath = Path.Combine(directory, "symlink.zip");
            using (var file = new FileStream(zipPath, FileMode.CreateNew))
            using (var zip = new ZipArchive(file, ZipArchiveMode.Create))
            {
                using (Stream stream = zip.CreateEntry("setup.ps1").Open()) stream.WriteByte(42);
                ZipArchiveEntry link = zip.CreateEntry("escape-link"); link.ExternalAttributes = unchecked((int)0xA1FF0000);
                using (Stream stream = link.Open()) stream.WriteByte(42);
            }
            Reject(() => Archive.ExtractVerifiedZip(zipPath, stage, 2, "setup.ps1", CancellationToken.None, Quiet), "UNIX symlink accepted.");
            Reject(() => Archive.ExtractVerifiedZip(Zip(directory, "setup.ps1", "assets", "assets/file"), stage, 3, "setup.ps1", CancellationToken.None, Quiet), "File/directory conflict accepted.");
            Reject(() => Archive.ExtractVerifiedZip(Zip(directory, "file.txt"), stage, 1, "setup.ps1", CancellationToken.None, Quiet), "Missing entry point accepted.");
            Reject(() => Archive.ExtractVerifiedZip(Zip(directory, "setup.ps1"), stage, 2, "setup.ps1", CancellationToken.None, Quiet), "Unpacked size mismatch accepted.");
            Assert(Directory.GetFileSystemEntries(stage).Length == 0, "Rejected metadata extracted files.");
        }
        private static void ReparseGuard()
        {
            string directory = Case("junction"), stage = Path.Combine(directory, "stage"), outside = Path.Combine(directory, "outside");
            Directory.CreateDirectory(stage); Directory.CreateDirectory(outside);
            string link = Path.Combine(stage, "assets"); CreateJunction(link, outside); junctions.Add(link);
            Reject(() => Safety.NoReparse(Path.Combine(link, "model.gguf")), "Reparse ancestor accepted.");
            Reject(() => Safety.NewDestination(Path.Combine(link, "new-install")), "Destination through reparse accepted.");
            string zip = Zip(directory, "setup.ps1", "assets/escape.txt");
            Reject(() => Archive.ExtractVerifiedZip(zip, stage, 2, "setup.ps1", CancellationToken.None, Quiet), "ZIP extraction followed reparse.");
            string source = Path.Combine(directory, "source.bin"); File.WriteAllBytes(source, Data(64));
            Reject(() => Download.Verified(new Source { FilePath = source }, link, 64, Download.HashFile(source, CancellationToken.None), "test", CancellationToken.None, Quiet), "Cache followed reparse.");
            Assert(Directory.GetFileSystemEntries(outside).Length == 0, "Reparse tests changed the destination outside staging.");
        }
        private static void NoOverwrite()
        {
            string directory = Case("no-overwrite"), stage = Path.Combine(directory, "existing"); Directory.CreateDirectory(stage);
            string sentinel = Path.Combine(stage, "setup.ps1"); File.WriteAllText(sentinel, "PRESERVE");
            Reject(() => Safety.NewDestination(stage), "Existing directory accepted.");
            Reject(() => Archive.ExtractVerifiedZip(Zip(directory, "setup.ps1"), stage, 1, "setup.ps1", CancellationToken.None, Quiet), "Existing file overwritten.");
            Assert(File.ReadAllText(sentinel) == "PRESERVE", "Existing installation changed.");
            string existingFile = Path.Combine(directory, "existing-file"); File.WriteAllText(existingFile, "PRESERVE");
            Reject(() => Safety.NewDestination(existingFile), "Existing file target accepted.");
            Reject(() => Safety.NewDestination(@"C:\LivingHome\forbidden-new-install"), "Existing runtime subtree accepted.");
            string ownerStage = Path.Combine(directory, "owned"); Directory.CreateDirectory(ownerStage); File.WriteAllText(Path.Combine(ownerStage, ".setup-owner"), "owner-A");
            Reject(() => Safety.DeleteOwnedStaging(ownerStage, "owner-B"), "Staging cleanup ignored ownership.");
            Assert(Directory.Exists(ownerStage), "Unowned staging was deleted.");
        }
        private static void DigestChecks()
        {
            string directory = Case("digest"), source = Path.Combine(directory, "source.bin"), cache = Path.Combine(directory, "cache");
            byte[] data = Data(32000); File.WriteAllBytes(source, data); string hash = Hash(data);
            Reject(() => Download.Verified(new Source { FilePath = source }, cache, data.Length + 1, hash, "test", CancellationToken.None, Quiet), "Wrong declared length accepted.");
            Reject(() => Download.Verified(new Source { FilePath = source }, cache, data.Length, new string('0', 64), "test", CancellationToken.None, Quiet), "Wrong SHA256 accepted.");
            Assert(!File.Exists(Path.Combine(cache, new string('0', 64) + ".verified")), "Bad digest published verified cache.");
            string cached = Download.Verified(new Source { FilePath = source }, cache, data.Length, hash, "test", CancellationToken.None, Quiet);
            File.WriteAllBytes(cached, Data(12));
            cached = Download.Verified(new Source { FilePath = source }, cache, data.Length, hash, "test", CancellationToken.None, Quiet);
            Download.Verify(cached, data.Length, hash, CancellationToken.None);
        }
        private static void CancellationResume()
        {
            string directory = Case("cancel"), source = Path.Combine(directory, "source.bin"), cache = Path.Combine(directory, "cache");
            byte[] data = Data(3 * 1024 * 1024 + 17); File.WriteAllBytes(source, data); string hash = Hash(data); bool cancelled = false;
            using (var stop = new CancellationTokenSource())
                try { Download.Verified(new Source { FilePath = source }, cache, data.Length, hash, "test", stop.Token, update => { if (update.Current > 0 && update.Current < data.Length) stop.Cancel(); }); }
                catch (OperationCanceledException) { cancelled = true; }
            string partial = Path.Combine(cache, hash + ".partial");
            Assert(cancelled && File.Exists(partial) && new FileInfo(partial).Length > 0 && new FileInfo(partial).Length < data.Length, "Cancellation did not retain a real partial file.");
            string complete = Download.Verified(new Source { FilePath = source }, cache, data.Length, hash, "test", CancellationToken.None, Quiet);
            Download.Verify(complete, data.Length, hash, CancellationToken.None);
        }
        private static void InterruptedHttp()
        {
            string directory = Case("http-interrupt"), cache = Path.Combine(directory, "cache"); byte[] data = Data(2 * 1024 * 1024 + 357);
            using (var server = new RangeServer(data, "interrupt"))
            {
                string complete = Download.Verified(new Source { RemoteUri = server.Uri }, cache, data.Length, Hash(data), "HTTP test", CancellationToken.None, Quiet);
                Download.Verify(complete, data.Length, Hash(data), CancellationToken.None);
                Assert(server.Requests >= 2 && server.RangeOffsets.Any(x => x > 0), "Interrupted HTTP did not request a resumable byte range.");
            }
        }
        private static void IgnoredRange()
        {
            string directory = Case("ignore-range"), cache = Path.Combine(directory, "cache"); Directory.CreateDirectory(cache); byte[] data = Data(8000); string hash = Hash(data);
            File.WriteAllBytes(Path.Combine(cache, hash + ".partial"), data.Take(500).ToArray());
            using (var server = new RangeServer(data, "ignore"))
            {
                string complete = Download.Verified(new Source { RemoteUri = server.Uri }, cache, data.Length, hash, "HTTP test", CancellationToken.None, Quiet);
                Download.Verify(complete, data.Length, hash, CancellationToken.None);
                Assert(server.RangeOffsets.Contains(500), "Range-ignore test did not attempt resume.");
            }
        }
        private static void InvalidRange()
        {
            string directory = Case("invalid-range"), cache = Path.Combine(directory, "cache"); Directory.CreateDirectory(cache); byte[] data = Data(8000); string hash = Hash(data);
            string partial = Path.Combine(cache, hash + ".partial"); File.WriteAllBytes(partial, data.Take(500).ToArray());
            using (var server = new RangeServer(data, "bad-range")) Reject(() => Download.Verified(new Source { RemoteUri = server.Uri }, cache, data.Length, hash, "HTTP test", CancellationToken.None, Quiet), "Invalid resume range accepted.");
            Assert(new FileInfo(partial).Length == 500, "Invalid range changed saved partial bytes.");
        }
        private static void WrongByteCount()
        {
            string directory = Case("wrong-http-size"); byte[] data = Data(8000);
            using (var server = new RangeServer(data, "oversize")) Reject(() => Download.Verified(new Source { RemoteUri = server.Uri }, Path.Combine(directory, "cache"), data.Length, Hash(data), "HTTP test", CancellationToken.None, Quiet), "Wrong Content-Length accepted.");
        }
        private static void BinaryCheckOnly()
        {
            Assert(!String.IsNullOrEmpty(bootstrapper) && File.Exists(bootstrapper), "Pass --bootstrapper to exercise the actual production executable.");
            string directory = Case("binary"), manifest = Path.Combine(directory, "release.json"), report = Path.Combine(directory, "check-report.json"), target = Path.Combine(directory, "new-install");
            ReleaseManifest release = Fixture(directory); File.WriteAllText(manifest, new JavaScriptSerializer().Serialize(release), new UTF8Encoding(false));
            string cache = Path.Combine(InstallerEngine.CacheRoot, Download.HashFile(manifest, CancellationToken.None));
            Assert(!Directory.Exists(cache), "Synthetic release unexpectedly already exists in cache.");
            int exit = RunBinary("--check-only --manifest \"" + manifest + "\" --install-dir \"" + target + "\" --report \"" + report + "\"");
            Assert(exit == 0 || exit == 3, "Check-only returned unexpected exit " + exit);
            Assert(File.Exists(report) && File.ReadAllText(report).Contains("manifestSha256"), "Production readiness report missing.");
            File.Copy(report, Path.Combine(Path.GetDirectoryName(AssemblyLocation()), "check-only-host.json"), true);
            Assert(!Directory.Exists(target) && !Directory.Exists(cache), "Check-only created installation or asset cache.");
            Assert(RunBinary("--unattended --manifest \"" + manifest + "\" --install-dir \"" + target + "\"") == 2, "Unattended missing-pin gate failed.");
            Assert(RunBinary("--check-only --manifest \"" + manifest + "\" --manifest-sha256 " + new string('0', 64) + " --install-dir \"" + target + "\"") == 2, "Production wrong-pin gate failed.");
            Assert(RunBinary("--check-only --manifest http://example.invalid/release.json --install-dir \"" + target + "\"") == 2, "Production HTTP manifest gate failed.");
            Assert(!Directory.Exists(target) && !Directory.Exists(cache), "Rejected CLI invocations created installation/cache.");
        }
        private static string AssemblyLocation() { return System.Reflection.Assembly.GetExecutingAssembly().Location; }
        private static int RunBinary(string arguments)
        {
            using (var process = Process.Start(new ProcessStartInfo(bootstrapper, arguments) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true }))
            {
                string output = process.StandardOutput.ReadToEnd(); string error = process.StandardError.ReadToEnd();
                if (!process.WaitForExit(60000)) { process.Kill(); throw new Exception("Production check-only timed out."); }
                if (output.Length > 0) Console.WriteLine(output.Trim()); if (error.Length > 0) Console.WriteLine(error.Trim()); return process.ExitCode;
            }
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern SafeFileHandle CreateFile(string name, uint access, uint share, IntPtr security, uint disposition, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool DeviceIoControl(SafeFileHandle handle, uint control, byte[] input, int inputSize, IntPtr output, int outputSize, out int returned, IntPtr overlapped);
        private static void CreateJunction(string link, string destination)
        {
            Safety.NoReparse(link); Safety.NoReparse(destination); Directory.CreateDirectory(link);
            byte[] substitute = Encoding.Unicode.GetBytes(@"\??\" + Path.GetFullPath(destination)); byte[] print = Encoding.Unicode.GetBytes(Path.GetFullPath(destination));
            byte[] buffer = new byte[16 + substitute.Length + 2 + print.Length + 2];
            Array.Copy(BitConverter.GetBytes(0xA0000003u), 0, buffer, 0, 4); Array.Copy(BitConverter.GetBytes((ushort)(buffer.Length - 8)), 0, buffer, 4, 2);
            Array.Copy(BitConverter.GetBytes((ushort)substitute.Length), 0, buffer, 10, 2); Array.Copy(BitConverter.GetBytes((ushort)(substitute.Length + 2)), 0, buffer, 12, 2); Array.Copy(BitConverter.GetBytes((ushort)print.Length), 0, buffer, 14, 2);
            Array.Copy(substitute, 0, buffer, 16, substitute.Length); Array.Copy(print, 0, buffer, 16 + substitute.Length + 2, print.Length);
            using (SafeFileHandle handle = CreateFile(link, 0x40000000, 0, IntPtr.Zero, 3, 0x02200000, IntPtr.Zero))
            {
                int returned;
                if (handle.IsInvalid || !DeviceIoControl(handle, 0x000900A4, buffer, buffer.Length, IntPtr.Zero, 0, out returned, IntPtr.Zero)) throw new Exception("Cannot create test junction, Windows error " + Marshal.GetLastWin32Error());
            }
        }
        private sealed class RangeServer : IDisposable
        {
            private readonly TcpListener listener; private readonly Thread thread; private readonly byte[] data; private readonly string mode; private volatile bool disposed;
            public int Requests; public List<long> RangeOffsets = new List<long>(); public Uri Uri;
            public RangeServer(byte[] bytes, string behavior)
            {
                data = bytes; mode = behavior; listener = new TcpListener(IPAddress.Loopback, 0); listener.Start();
                Uri = new Uri("http://127.0.0.1:" + ((IPEndPoint)listener.LocalEndpoint).Port + "/fixture.bin");
                thread = new Thread(Serve) { IsBackground = true }; thread.Start();
            }
            private void Serve()
            {
                try
                {
                    while (!disposed)
                    {
                        using (TcpClient client = listener.AcceptTcpClient())
                        using (NetworkStream stream = client.GetStream())
                        {
                            client.ReceiveTimeout = 10000; client.SendTimeout = 10000;
                            var headers = new StringBuilder();
                            while (headers.Length < 16384 && !headers.ToString().EndsWith("\r\n\r\n")) { int next = stream.ReadByte(); if (next < 0) break; headers.Append((char)next); }
                            Match match = Regex.Match(headers.ToString(), @"Range: bytes=(\d+)-", RegexOptions.IgnoreCase); long offset = match.Success ? Int64.Parse(match.Groups[1].Value) : 0;
                            Requests++; RangeOffsets.Add(offset);
                            bool partial = offset > 0 && mode != "ignore";
                            long sendOffset = partial ? offset : 0, length = data.Length - sendOffset;
                            string header = "HTTP/1.1 " + (partial ? "206 Partial Content" : "200 OK") + "\r\nConnection: close\r\nContent-Length: " + (mode == "oversize" ? length + 1 : length) + "\r\n";
                            if (partial) header += "Content-Range: bytes " + (mode == "bad-range" ? offset + 1 : offset) + "-" + (data.Length - 1) + "/" + data.Length + "\r\n";
                            byte[] headerBytes = Encoding.ASCII.GetBytes(header + "\r\n"); stream.Write(headerBytes, 0, headerBytes.Length);
                            int count = (int)length;
                            if (mode == "interrupt" && Requests == 1) count = 1024 * 1024 + 17;
                            stream.Write(data, (int)sendOffset, count); stream.Flush();
                        }
                    }
                }
                catch (SocketException) { if (!disposed) throw; }
                catch (IOException) { if (!disposed) throw; }
            }
            public void Dispose() { disposed = true; listener.Stop(); thread.Join(10000); }
        }
    }
}
