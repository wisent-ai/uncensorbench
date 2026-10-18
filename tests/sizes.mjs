#!/usr/bin/env node
// Lists the text files of one committed revision that hold more lines than a
// limit the caller states, for a defect register's revision-bound completion
// probe of files that must be split:
//
//   node tests/sizes.mjs <revision> <line-limit> <path>...
//
// Standard output carries one `revision:path:lines` per file over the limit
// and is empty when none is. Lines are counted by git grep, which reads every
// pathspec form git accepts (globs and exclusions included) and leaves binary
// files out. A revision or pattern git refuses fails the command with git's
// reason.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
// argv holds node and this script before the caller's words.
const [, , revision, stated, ...paths] = process.argv;
assert.ok(revision && stated && paths.length, 'usage: node tests/sizes.mjs <revision> <line-limit> <path>...');
const limit = Number(stated);
assert.ok(Number.isSafeInteger(limit), `line-limit must be a whole number, not ${stated}`);

// An empty pattern matches every line, so -c answers each file's line count.
const counted = spawnSync('git', ['-C', root, 'grep', '-c', '-I', '-e', '', revision, '--', ...paths], {
  encoding: 'utf8',
  maxBuffer: Infinity,
});
// git grep answers 1 when nothing matched and above 1 when it failed
// (https://git-scm.com/docs/git-grep#_exit_status).
const matched = spawnSync('true').status;
const nothing = spawnSync('false').status;
assert.ok(counted.status === matched || counted.status === nothing,
  `git grep at ${revision} failed (${counted.status}${counted.error ? `, ${counted.error.message}` : ''}): ${counted.stderr}`);
// Each row is `revision:path:count`; the count is the digits after the last colon.
for (const row of counted.stdout.split('\n').filter(Boolean)) {
  const { count } = row.match(/:(?<count>\d+)$/).groups;
  if (Number(count) > limit) process.stdout.write(`${row}\n`);
}
