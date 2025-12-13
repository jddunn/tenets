import { ChildProcess, spawn } from 'child_process';
import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class TenetsServerManager {
  private process?: ChildProcess;
  private outputChannel: vscode.OutputChannel;
  private statusBarItem: vscode.StatusBarItem;
  private isRunning = false;

  constructor() {
    this.outputChannel = vscode.window.createOutputChannel('Tenets MCP');
    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.statusBarItem.command = 'tenets.viewLogs';
    this.updateStatusBar();
    this.statusBarItem.show();
  }

  private updateStatusBar() {
    if (this.isRunning) {
      this.statusBarItem.text = '$(check) Tenets: Active';
      this.statusBarItem.tooltip = 'Tenets MCP Server is running. Click to view logs.';
      this.statusBarItem.backgroundColor = undefined;
    } else {
      this.statusBarItem.text = '$(circle-slash) Tenets: Inactive';
      this.statusBarItem.tooltip = 'Tenets MCP Server is not running. Click to view logs.';
      this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
    }
  }

  async findTenetsMcp(): Promise<string | null> {
    const config = vscode.workspace.getConfiguration('tenets');
    const configuredPath = config.get<string>('mcpPath');

    if (configuredPath) {
      this.outputChannel.appendLine(`Using configured path: ${configuredPath}`);
      return configuredPath;
    }

    // Try common locations
    const locations = [
      '/Users/johnn/.local/bin/tenets-mcp',  // pipx default
      '~/.local/bin/tenets-mcp',
      'tenets-mcp',  // PATH
    ];

    for (const location of locations) {
      try {
        const { stdout } = await execAsync(`which ${location}`);
        const path = stdout.trim();
        if (path) {
          this.outputChannel.appendLine(`Found tenets-mcp at: ${path}`);
          return path;
        }
      } catch {
        // Continue trying other locations
      }
    }

    // Try 'which tenets-mcp' as last resort
    try {
      const { stdout } = await execAsync('which tenets-mcp');
      const path = stdout.trim();
      if (path) {
        this.outputChannel.appendLine(`Found tenets-mcp at: ${path}`);
        return path;
      }
    } catch {
      // Not found
    }

    this.outputChannel.appendLine('Could not find tenets-mcp executable');
    return null;
  }

  async start() {
    if (this.isRunning) {
      this.outputChannel.appendLine('Server is already running');
      return;
    }

    const tenetsPath = await this.findTenetsMcp();
    if (!tenetsPath) {
      vscode.window.showErrorMessage(
        'Tenets MCP Server not found. Please install it with: pipx install tenets[mcp]'
      );
      return;
    }

    this.outputChannel.appendLine(`Starting Tenets MCP Server: ${tenetsPath}`);
    this.outputChannel.show(true);

    try {
      this.process = spawn(tenetsPath, [], { shell: true });

      this.process.stdout?.on('data', (data) => {
        this.outputChannel.append(data.toString());
      });

      this.process.stderr?.on('data', (data) => {
        this.outputChannel.append(`[ERROR] ${data.toString()}`);
      });

      this.process.on('close', (code) => {
        this.outputChannel.appendLine(`Server exited with code ${code}`);
        this.isRunning = false;
        this.updateStatusBar();
      });

      this.process.on('error', (error) => {
        this.outputChannel.appendLine(`Failed to start server: ${error.message}`);
        vscode.window.showErrorMessage(`Failed to start Tenets MCP Server: ${error.message}`);
        this.isRunning = false;
        this.updateStatusBar();
      });

      this.isRunning = true;
      this.updateStatusBar();
      vscode.window.showInformationMessage('Tenets MCP Server started');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.outputChannel.appendLine(`Error starting server: ${message}`);
      vscode.window.showErrorMessage(`Failed to start Tenets MCP Server: ${message}`);
    }
  }

  async stop() {
    if (!this.isRunning || !this.process) {
      this.outputChannel.appendLine('Server is not running');
      return;
    }

    this.outputChannel.appendLine('Stopping Tenets MCP Server...');
    this.process.kill('SIGTERM');

    // Give it 5 seconds to gracefully shutdown
    setTimeout(() => {
      if (this.process && !this.process.killed) {
        this.outputChannel.appendLine('Force killing server...');
        this.process.kill('SIGKILL');
      }
    }, 5000);

    this.isRunning = false;
    this.process = undefined;
    this.updateStatusBar();
    vscode.window.showInformationMessage('Tenets MCP Server stopped');
  }

  async restart() {
    await this.stop();
    // Wait a bit before restarting
    setTimeout(() => {
      this.start();
    }, 1000);
  }

  showLogs() {
    this.outputChannel.show();
  }

  dispose() {
    this.stop();
    this.outputChannel.dispose();
    this.statusBarItem.dispose();
  }
}
