import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
// import path from 'path'; // Removed as per ESLint fix

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export async function GET(_request: Request) { // Renamed request to _request
  // Construct the path to app.py relative to the current file
  // Current file: webapp/src/app/api/run-strategy/route.ts
  // Target file: app.py (at the root of the repository)
  // Path: ../../../../app.py
  // const scriptPath = path.join(__dirname, '../../../../../app.py'); // Unused, __dirname behavior varies.
  // Note: In Next.js deployed environments, __dirname might behave differently.
  // For local dev, it's usually fine. A more robust way in Next.js might be:
  // const scriptPath = path.resolve(process.cwd(), '../../app.py'); if API route is in /app
  // Or, if CWD is /app/webapp:
  // const scriptPath = path.resolve(process.cwd(), '../app.py'); 
  // For this specific structure, assuming CWD is /app (repo root) for the spawn, or use absolute path.
  // Let's assume the execution context of spawn will be the repo root or adjust scriptPath.
  
  // Safest in this sandbox/Docker might be to use an absolute path from /app/
  const scriptToExecute = '/app/app.py'; // Using this as the definitive path


  return new Promise((resolve) => {
    const pythonProcess = spawn('python3', [scriptToExecute, '--json-output']); // Used scriptToExecute
    
    let stdout_data = '';
    let stderr_data = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout_data += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr_data += data.toString();
    });

    pythonProcess.on('error', (error) => {
      console.error('Spawn error:', error);
      resolve(NextResponse.json({ error: 'Failed to start strategy script', details: error.message }, { status: 500 }));
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          // Attempt to parse stdout as JSON
          // If app.py doesn't produce valid JSON yet, this will fail.
          // For now, we'll assume it might, or return raw output.
          // The task specifies "if the script executes successfully and produces JSON on stdout, parse the JSON"
          // This implies we should attempt to parse.
          const jsonData = JSON.parse(stdout_data);
          resolve(NextResponse.json(jsonData));
        } catch (e) {
          console.error('JSON parsing error:', e);
          // If JSON parsing fails, but script exited with 0, it means output was not valid JSON.
          // Return the raw stdout, or a specific error message.
          resolve(NextResponse.json({ error: 'Strategy script ran but output was not valid JSON', details: stdout_data || 'No output' }, { status: 500 }));
        }
      } else {
        console.error(`Script exited with code ${code}: ${stderr_data}`);
        resolve(NextResponse.json({ error: 'Failed to run strategy', details: stderr_data || stdout_data || `Script exited with code ${code}` }, { status: 500 }));
      }
    });
  });
}
