import { createContext, useContext, useState, useEffect } from 'react';
import {
  onAuthStateChanged,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  sendPasswordResetEmail,
  sendEmailVerification,
  reload,
} from 'firebase/auth';
import { auth } from '../config/firebase';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [emailVerified, setEmailVerified] = useState(false);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser);
      setEmailVerified(Boolean(firebaseUser?.emailVerified));
      setLoading(false);
    });
    return unsubscribe;
  }, []);

  const signIn = (email, password) =>
    signInWithEmailAndPassword(auth, email, password);

  const signUp = async (email, password) => {
    const cred = await createUserWithEmailAndPassword(auth, email, password);
    // Kick off verification email immediately on account creation.
    try {
      await sendEmailVerification(cred.user);
    } catch {
      // Non-fatal; user can resend from the banner.
    }
    return cred;
  };

  const logOut = () => signOut(auth);

  const resetPassword = (email) => sendPasswordResetEmail(auth, email);

  const sendVerification = async () => {
    if (!auth.currentUser) return;
    await sendEmailVerification(auth.currentUser);
  };

  const refreshVerification = async () => {
    if (!auth.currentUser) return false;
    await reload(auth.currentUser);
    const verified = Boolean(auth.currentUser.emailVerified);
    setEmailVerified(verified);
    // Trigger a user ref update so consumers re-render with fresh emailVerified.
    setUser({ ...auth.currentUser });
    return verified;
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        emailVerified,
        signIn,
        signUp,
        logOut,
        resetPassword,
        sendVerification,
        refreshVerification,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
