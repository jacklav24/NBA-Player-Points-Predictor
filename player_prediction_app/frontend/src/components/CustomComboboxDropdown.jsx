import { Combobox, ComboboxInput, ComboboxOption, ComboboxOptions, ComboboxButton } from '@headlessui/react';
import { CheckIcon, ChevronDownIcon } from '@heroicons/react/24/solid'
import { useState, useEffect } from 'react';



export default function CustomComboboxDropdown({ label, options, value, onChange, disabled = false, displayMap = {}}) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);

  const shortLabel = (val) => val;
  const longLabel = (val) => displayMap[val] || val;
  const filtered =
    query === ''
      ? options
      : options.filter(opt => longLabel(opt).toLowerCase().includes(query.toLowerCase()))
    return (
      <div className="flex-1 min-w-[150px] max-w-[200px]">
        <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
        <Combobox value={value} onChange={onChange} disabled={disabled}>
          <div className="relative">
            <ComboboxInput
              className="w-full px-3 py-2 rounded border border-gray-600 bg-[#1e2147] text-[#f5f5f5] focus:outline-none focus:ring-2 focus:ring-indigo-400"
              displayValue={shortLabel}
              onChange={e => setQuery(e.target.value)}
              onFocus={() => setOpen(true)}
              onBlur={() => setTimeout(() => setOpen(false), 100)}
              placeholder={`Select ${label}`}
            />
            <ComboboxButton className="absolute inset-y-0 right-0 flex items-center px-2.5">
              <ChevronDownIcon className="h-5 w-5 text-white/60 group-hover:text-white" />
            </ComboboxButton>
  
            {open && filtered.length > 0 && (
              <ComboboxOptions className="absolute mt-1 w-full max-h-60 overflow-auto rounded-md bg-[#2a2d55] py-1 text-sm shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none z-50">
                {filtered.map(opt => (
                  <ComboboxOption
                    key={opt}
                    value={opt}
                    className="cursor-pointer select-none px-4 py-2 ui-active:bg-indigo-600 ui-active:text-white text-[#f5f5f5]"
                  >
                    {longLabel(opt)}
                  </ComboboxOption>
                ))}
              </ComboboxOptions>
            )}
          </div>
        </Combobox>
      </div>
    );
  }
  
