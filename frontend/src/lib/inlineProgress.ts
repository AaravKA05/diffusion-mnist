/*
 * Tracks whether the page is already showing progress next to the work
 * (the Studio's "Step 12 of 20", the launch-page demo). While it is, the
 * floating ActivityToast stays hidden so progress is not shown twice.
 */

let count = 0;
const listeners = new Set<(showing: boolean) => void>();

function emit() {
  listeners.forEach((fn) => fn(count > 0));
}

/** Call while inline progress is on screen; returns the release function. */
export function claimInlineProgress() {
  count += 1;
  emit();
  let released = false;
  return () => {
    if (released) return;
    released = true;
    count -= 1;
    emit();
  };
}

export function onInlineProgressChange(fn: (showing: boolean) => void) {
  listeners.add(fn);
  fn(count > 0);
  return () => {
    listeners.delete(fn);
  };
}
