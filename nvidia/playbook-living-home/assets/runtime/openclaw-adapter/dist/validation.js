                                       
export class ContractError extends Error {}

// Structural checks are followed by operation-specific checks in tools.ts.
export function validate(schema      , value     )       {
  if (schema.enum && !schema.enum.includes(value)) throw new ContractError('Unsupported argument choice');
  if (schema.oneOf) {
    let matches = 0;
    for (const variant of schema.oneOf) {
      try { validate(variant, value); matches++; } catch (error) { if (!(error instanceof ContractError)) throw error; }
    }
    if (matches !== 1) throw new ContractError('Argument must match exactly one supported shape');
  }
  if (schema.type === 'object') {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new ContractError('Expected an argument object');
    for (const key of schema.required || []) if (!Object.hasOwn(value, key)) throw new ContractError('A required argument is missing');
    for (const [key, child] of Object.entries(value)) {
      if (Object.hasOwn(schema.properties || {}, key)) validate(schema.properties[key], child);
      else if (schema.additionalProperties === false) throw new ContractError('Unsupported argument field');
    }
  } else if (schema.type === 'array') {
    if (!Array.isArray(value) || value.length < (schema.minItems ?? 0) || value.length > (schema.maxItems ?? Infinity))
      throw new ContractError('Argument list length is outside the supported range');
    for (const item of value) validate(schema.items || {}, item);
  } else if (schema.type === 'string') {
    if (typeof value !== 'string' || value.length < (schema.minLength ?? 0) || value.length > (schema.maxLength ?? Infinity)
        || (schema.pattern && !(new RegExp(schema.pattern)).test(value))) throw new ContractError('Argument text is invalid');
  } else if (schema.type === 'boolean') {
    if (typeof value !== 'boolean') throw new ContractError('Expected a boolean argument');
  } else if (schema.type === 'integer' || schema.type === 'number') {
    if (typeof value !== 'number' || !Number.isFinite(value) || (schema.type === 'integer' && !Number.isInteger(value))
        || value < (schema.minimum ?? -Infinity) || value > (schema.maximum ?? Infinity))
      throw new ContractError('Argument number is outside the supported range');
  }
}

export function fields(value      , allowed          )       {
  if (Object.keys(value).some(key => !allowed.includes(key))) throw new ContractError('Unsupported operation field');
}
export function requiredText(value      , ...names          )       {
  if (names.some(name => typeof value[name] !== 'string' || !value[name].trim()))
    throw new ContractError('Supply all required non-empty text fields');
}
export function exactId(value         , allowCurrent = false)         {
  if (typeof value !== 'string' || !/^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}$/.test(value)
      || ['.', '..'].includes(value) || (!allowCurrent && value === 'current'))
    throw new ContractError('Use the exact observed record ID, not a path or URL');
  return value;
}
export function pick(value      , allowed          )       {
  return Object.fromEntries(allowed.filter(key => Object.hasOwn(value, key)).map(key => [key, value[key]]));
}

export function redact(value     , token        )      {
  if (typeof value === 'string') return token ? value.replaceAll(token, '[redacted]') : value;
  if (Array.isArray(value)) return value.map(item => redact(item, token));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, redact(item, token)]));
  return value;
}


//# sourceURL=validation.ts