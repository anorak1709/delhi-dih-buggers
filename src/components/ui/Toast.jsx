import { useApp } from '../../context/AppContext';
import { motion, AnimatePresence } from 'framer-motion';

const icons = {
  success: (
    <svg className="w-4 h-4 text-up" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  ),
  error: (
    <svg className="w-4 h-4 text-down" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
  info: (
    <svg className="w-4 h-4 text-info" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" />
    </svg>
  ),
  warning: (
    <svg className="w-4 h-4 text-warn" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01M10.29 3.86l-8.6 14.86A1 1 0 002.56 20h18.88a1 1 0 00.87-1.28l-8.6-14.86a1 1 0 00-1.72 0z" />
    </svg>
  ),
};

const toastBorder = {
  success: 'border-l-up',
  error: 'border-l-down',
  info: 'border-l-info',
  warning: 'border-l-warn',
};

export default function ToastContainer() {
  const { toasts, dismissToast } = useApp();

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
      <AnimatePresence mode="popLayout">
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            layout
            initial={{ opacity: 0, x: 80, scale: 0.85, filter: 'blur(4px)' }}
            animate={{ opacity: 1, x: 0, scale: 1, filter: 'blur(0px)' }}
            exit={{ opacity: 0, x: 80, scale: 0.9, filter: 'blur(2px)' }}
            transition={{ type: 'spring', stiffness: 350, damping: 25 }}
            whileHover={{ scale: 1.02, x: -4 }}
            className={`flex items-center gap-3 rounded-lg border border-surface-700/80 dark:border-surface-700/80 border-surface-200 border-l-2 ${toastBorder[t.type] || 'border-l-info'} bg-surface-800/90 dark:bg-surface-800/90 bg-white backdrop-blur-md px-4 py-3 shadow-xl max-w-sm cursor-pointer`}
            onClick={() => dismissToast(t.id)}
          >
            {icons[t.type] || icons.info}
            <span className="text-sm text-surface-200 dark:text-surface-200 text-surface-700">{t.message}</span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
