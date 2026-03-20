import { motion } from 'framer-motion';

export function CardHeader({ title, subtitle, action, className = '' }) {
  return (
    <div className={`flex items-center justify-between mb-5 ${className}`}>
      <div>
        <h3 className="font-display text-lg font-semibold text-surface-50 dark:text-surface-50 text-surface-900 tracking-wide">
          {title}
        </h3>
        {subtitle && (
          <p className="text-xs text-surface-500 mt-1 leading-relaxed">{subtitle}</p>
        )}
      </div>
      {action && <div>{action}</div>}
    </div>
  );
}

export default function Card({ children, hover = true, className = '', delay = 0, onClick, ...props }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16, filter: 'blur(6px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      transition={{ type: 'spring', stiffness: 260, damping: 22, delay }}
      whileHover={hover ? { y: -3, transition: { type: 'spring', stiffness: 400, damping: 20 } } : {}}
      onClick={onClick}
      className={`
        glass-card gradient-border
        rounded-2xl p-5
        transition-all duration-500
        ${onClick ? 'cursor-pointer' : ''}
        ${className}
      `.replace(/\n\s+/g, ' ').trim()}
      {...props}
    >
      {children}
    </motion.div>
  );
}
