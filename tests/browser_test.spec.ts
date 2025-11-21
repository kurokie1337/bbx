import { test, expect } from '@playwright/test';

test('test workflow creation', async ({ page }) => {
  // 1. Open browser
  await page.goto('http://localhost:5173/builder');

  // 2. Test workflow creation
  await page.getByRole('button', { name: 'New Workflow' }).click();
  
  await page.getByPlaceholder('Workflow Name').click();
  await page.getByPlaceholder('Workflow Name').fill('Test Automation');
  
  await page.getByRole('button', { name: 'Save' }).click();

  // 3. Verify message
  await expect(page.getByText('Workflow saved!')).toBeVisible();

  // 4. Take screenshot
  await page.screenshot({ path: 'artifacts/screenshots/workflow-saved.png' });
});
