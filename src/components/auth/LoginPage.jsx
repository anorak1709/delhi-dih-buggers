import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import Button from '../ui/Button';
import Input from '../ui/Input';

const MODE = {
  SIGNIN: 'signin',
  SIGNUP: 'signup',
  RESET: 'reset',
};

function friendlyError(err) {
  const code = err?.code || '';
  if (
    code === 'auth/user-not-found' ||
    code === 'auth/wrong-password' ||
    code === 'auth/invalid-credential'
  ) {
    return 'Invalid email or password.';
  }
  if (code === 'auth/email-already-in-use') {
    return 'An account with this email already exists.';
  }
  if (code === 'auth/weak-password') {
    return 'Password should be at least 6 characters.';
  }
  if (code === 'auth/invalid-email') {
    return 'Please enter a valid email address.';
  }
  if (code === 'auth/too-many-requests') {
    return 'Too many attempts. Please wait a moment and try again.';
  }
  if (code === 'auth/missing-email') {
    return 'Please enter your email address.';
  }
  if (code === 'auth/network-request-failed') {
    return 'Network error. Check your connection and try again.';
  }
  return err?.message || 'Something went wrong.';
}

export default function LoginPage() {
  const { signIn, signUp, resetPassword } = useAuth();
  const [mode, setMode] = useState(MODE.SIGNIN);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');
  const [loading, setLoading] = useState(false);
  const [justSignedUp, setJustSignedUp] = useState(false);

  const switchMode = (next) => {
    setMode(next);
    setError('');
    setInfo('');
    setJustSignedUp(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setInfo('');
    setLoading(true);
    try {
      if (mode === MODE.SIGNUP) {
        await signUp(email, password);
        setJustSignedUp(true);
        setInfo(`Account created. We sent a verification link to ${email}.`);
      } else if (mode === MODE.RESET) {
        await resetPassword(email);
        setInfo(`If an account exists for ${email}, a reset link is on its way.`);
      } else {
        await signIn(email, password);
      }
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  const title =
    mode === MODE.SIGNUP
      ? 'Create Account'
      : mode === MODE.RESET
      ? 'Reset Password'
      : 'Welcome Back';

  const submitLabel =
    mode === MODE.SIGNUP
      ? 'Create Account'
      : mode === MODE.RESET
      ? 'Send Reset Link'
      : 'Sign In';

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface-900 dark:bg-surface-900 bg-surface-50 relative overflow-hidden">
      <div className="grain-overlay" />

      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 rounded-full bg-accent/5 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 rounded-full bg-accent/3 blur-3xl" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30, filter: 'blur(8px)' }}
        animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
        transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
        className="relative z-10 w-full max-w-md px-6"
      >
        {/* Brand */}
        <div className="text-center mb-8">
          <motion.h1
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="font-display text-4xl font-semibold text-gradient mb-2"
          >
            Bloom
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.35 }}
            className="text-xs uppercase tracking-[0.25em] text-surface-500"
          >
            Portfolio Intelligence
          </motion.p>
        </div>

        {/* Card */}
        <motion.div
          layout
          className="rounded-xl border border-surface-700/80 dark:border-surface-700/80 border-surface-200 bg-surface-800/60 dark:bg-surface-800/60 bg-white backdrop-blur-sm p-8"
        >
          <AnimatePresence mode="wait">
            <motion.h2
              key={mode}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.2 }}
              className="text-lg font-semibold text-surface-100 dark:text-surface-100 text-surface-800 mb-2"
            >
              {title}
            </motion.h2>
          </AnimatePresence>

          {mode === MODE.RESET && (
            <p className="text-xs text-surface-400 mb-6">
              Enter the email on your account and we'll send a secure link to reset your password.
            </p>
          )}
          {mode !== MODE.RESET && <div className="mb-4" />}

          <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              label="Email"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
            {mode !== MODE.RESET && (
              <Input
                label="Password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                autoComplete={mode === MODE.SIGNUP ? 'new-password' : 'current-password'}
              />
            )}

            {mode === MODE.SIGNIN && (
              <div className="flex justify-end -mt-1">
                <button
                  type="button"
                  onClick={() => switchMode(MODE.RESET)}
                  className="text-xs text-surface-400 hover:text-accent transition-colors duration-200 cursor-pointer"
                >
                  Forgot password?
                </button>
              </div>
            )}

            {error && (
              <motion.p
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-xs text-down"
              >
                {error}
              </motion.p>
            )}

            {info && (
              <motion.p
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-xs text-accent"
              >
                {info}
              </motion.p>
            )}

            <Button
              type="submit"
              loading={loading}
              className="w-full mt-2"
              size="lg"
              disabled={justSignedUp && mode === MODE.SIGNUP}
            >
              {submitLabel}
            </Button>
          </form>

          <div className="mt-6 text-center space-y-2">
            {mode === MODE.RESET ? (
              <button
                type="button"
                onClick={() => switchMode(MODE.SIGNIN)}
                className="text-xs text-surface-400 hover:text-accent transition-colors duration-200 cursor-pointer"
              >
                Back to sign in
              </button>
            ) : (
              <button
                type="button"
                onClick={() =>
                  switchMode(mode === MODE.SIGNUP ? MODE.SIGNIN : MODE.SIGNUP)
                }
                className="text-xs text-surface-400 hover:text-accent transition-colors duration-200 cursor-pointer"
              >
                {mode === MODE.SIGNUP
                  ? 'Already have an account? Sign in'
                  : "Don't have an account? Create one"}
              </button>
            )}
          </div>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-center text-2xs text-surface-600 mt-6"
        >
          Bloom Analytics v1.0
        </motion.p>
      </motion.div>
    </div>
  );
}
