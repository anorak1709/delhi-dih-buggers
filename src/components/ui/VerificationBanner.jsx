import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import { useApp } from '../../context/AppContext';

export default function VerificationBanner() {
  const { user, sendVerification, refreshVerification } = useAuth();
  const { notify } = useApp();
  const [sending, setSending] = useState(false);
  const [checking, setChecking] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  if (!user || user.emailVerified || dismissed) return null;

  const handleResend = async () => {
    setSending(true);
    try {
      await sendVerification();
      notify('Verification email sent. Check your inbox.', 'success');
    } catch (err) {
      const code = err?.code || '';
      const msg =
        code === 'auth/too-many-requests'
          ? 'Too many requests. Please wait before resending.'
          : 'Could not send verification email. Please try again.';
      notify(msg, 'error');
    } finally {
      setSending(false);
    }
  };

  const handleCheck = async () => {
    setChecking(true);
    try {
      const verified = await refreshVerification();
      if (verified) {
        notify('Email verified. Thanks!', 'success');
      } else {
        notify('Not verified yet. Click the link in the email we sent.', 'info');
      }
    } finally {
      setChecking(false);
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.25 }}
        className="mx-6 lg:mx-8 mt-4 rounded-lg border-l-4 border-accent glass-subtle px-4 py-3 flex flex-col sm:flex-row sm:items-center gap-3"
        role="status"
      >
        <div className="flex-1 text-sm text-surface-200 dark:text-surface-200 text-surface-700">
          <span className="font-medium text-accent">Verify your email.</span>{' '}
          We sent a confirmation link to{' '}
          <span className="font-mono text-xs">{user.email}</span>. Some features
          may be limited until you verify.
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleCheck}
            disabled={checking}
            className="text-xs px-3 py-1.5 rounded-md border border-surface-600/60 text-surface-200 dark:text-surface-200 text-surface-700 hover:border-accent hover:text-accent transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {checking ? 'Checking…' : "I've verified"}
          </button>
          <button
            type="button"
            onClick={handleResend}
            disabled={sending}
            className="text-xs px-3 py-1.5 rounded-md bg-accent/10 text-accent hover:bg-accent/20 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {sending ? 'Sending…' : 'Resend email'}
          </button>
          <button
            type="button"
            aria-label="Dismiss"
            onClick={() => setDismissed(true)}
            className="text-xs w-6 h-6 flex items-center justify-center rounded-md text-surface-500 hover:text-surface-200 transition-colors"
          >
            ×
          </button>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
