@echo off
REM Marine Trace AI - public demo link (Cloudflare quick tunnel, free, no account)
REM 1. Start the app first (start.bat).  2. Double-click this file.
REM 3. Copy the https://....trycloudflare.com address it prints and open it on any phone or laptop.
REM The link works only while this window and the app are running. Close this window to stop sharing.
set PORT=%1
if "%PORT%"=="" set PORT=5173
if not exist D:\Tools\cloudflared.exe (
  echo cloudflared not found in D:\Tools. Download it first.
  pause
  exit /b 1
)
echo Sharing http://localhost:%PORT% ...
echo Look below for the line with  https://....trycloudflare.com
D:\Tools\cloudflared.exe tunnel --url http://127.0.0.1:%PORT%
