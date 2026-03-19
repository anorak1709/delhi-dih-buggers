import { motion, AnimatePresence } from 'framer-motion';

// ── Stagger container + child ──────────────────────────────────
export const stagger = {
  container: {
    hidden: {},
    show: {
      transition: { staggerChildren: 0.06, delayChildren: 0.04 },
    },
  },
  item: {
    hidden: { opacity: 0, y: 16, filter: 'blur(4px)' },
    show: {
      opacity: 1,
      y: 0,
      filter: 'blur(0px)',
      transition: { type: 'spring', stiffness: 300, damping: 24 },
    },
  },
};

// ── Panel page transition ──────────────────────────────────────
export const pageTransition = {
  initial: { opacity: 0, y: 20, filter: 'blur(6px)' },
  animate: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] },
  },
  exit: {
    opacity: 0,
    y: -12,
    filter: 'blur(4px)',
    transition: { duration: 0.25, ease: [0.55, 0.06, 0.68, 0.19] },
  },
};

// ── Fade in from bottom (cards, sections) ──────────────────────
export const fadeUp = {
  initial: { opacity: 0, y: 24 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { type: 'spring', stiffness: 260, damping: 20 },
  },
};

// ── Scale tap for buttons ──────────────────────────────────────
export const tapScale = {
  whileTap: { scale: 0.96 },
  whileHover: { scale: 1.02 },
  transition: { type: 'spring', stiffness: 400, damping: 17 },
};

// ── Toast slide in/out ─────────────────────────────────────────
export const toastVariants = {
  initial: { opacity: 0, x: 80, scale: 0.85 },
  animate: {
    opacity: 1,
    x: 0,
    scale: 1,
    transition: { type: 'spring', stiffness: 350, damping: 25 },
  },
  exit: {
    opacity: 0,
    x: 80,
    scale: 0.9,
    transition: { duration: 0.2, ease: 'easeIn' },
  },
};

// ── Row appear (for table rows, list items) ────────────────────
export const rowVariants = {
  hidden: { opacity: 0, x: -12 },
  visible: (i) => ({
    opacity: 1,
    x: 0,
    transition: {
      delay: i * 0.04,
      type: 'spring',
      stiffness: 300,
      damping: 24,
    },
  }),
  exit: {
    opacity: 0,
    x: 20,
    height: 0,
    transition: { duration: 0.2 },
  },
};

// ── Reusable Stagger wrapper ───────────────────────────────────
export function StaggerList({ children, className = '' }) {
  return (
    <motion.div
      variants={stagger.container}
      initial="hidden"
      animate="show"
      className={className}
    >
      {children}
    </motion.div>
  );
}

export function StaggerItem({ children, className = '' }) {
  return (
    <motion.div variants={stagger.item} className={className}>
      {children}
    </motion.div>
  );
}

// ── Number counter animation ───────────────────────────────────
export { motion, AnimatePresence };
