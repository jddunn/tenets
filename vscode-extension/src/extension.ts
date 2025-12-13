import * as vscode from 'vscode';
import { TenetsServerManager } from './serverManager';

let serverManager: TenetsServerManager | undefined;

export async function activate(context: vscode.ExtensionContext) {
  console.log('Tenets MCP Server extension activated');

  // Create server manager
  serverManager = new TenetsServerManager();

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('tenets.start', async () => {
      await serverManager?.start();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('tenets.stop', async () => {
      await serverManager?.stop();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('tenets.restart', async () => {
      await serverManager?.restart();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('tenets.viewLogs', () => {
      serverManager?.showLogs();
    })
  );

  // Auto-start if configured
  const config = vscode.workspace.getConfiguration('tenets');
  const autoStart = config.get<boolean>('autoStart', true);

  if (autoStart) {
    // Delay auto-start slightly to avoid overwhelming startup
    setTimeout(async () => {
      await serverManager?.start();
    }, 2000);
  }

  // Cleanup on disposal
  context.subscriptions.push({
    dispose: () => {
      serverManager?.dispose();
    }
  });
}

export function deactivate() {
  serverManager?.dispose();
}
