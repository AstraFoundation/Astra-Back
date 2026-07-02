import { describe, expect, it } from "vitest";

import { AstraApiError, HttpSession } from "../src/http.js";
import { installFetch } from "./_mock.js";

describe("HttpSession", () => {
  it("retries transient 503s and then succeeds", async () => {
    let n = 0;
    installFetch(() => {
      n += 1;
      return n < 3 ? { status: 503, json: {} } : { json: { ok: true } };
    });
    const s = new HttpSession("http://t", "k", { maxAttempts: 3, maxBackoff: 0 });
    const resp = await s.request("GET", "/x");
    expect(await resp.json()).toEqual({ ok: true });
    expect(n).toBe(3);
    s.close();
  });

  it("does not retry a 401 and raises AstraApiError", async () => {
    let n = 0;
    installFetch(() => {
      n += 1;
      return {
        status: 401,
        json: { detail: { code: "invalid_api_key", message: "bad" } },
      };
    });
    const s = new HttpSession("http://t", "k", { maxAttempts: 3, maxBackoff: 0 });
    await expect(s.request("GET", "/x")).rejects.toMatchObject({
      name: "AstraApiError",
      code: "invalid_api_key",
      status: 401,
    });
    expect(n).toBe(1);
    expect(AstraApiError).toBeDefined();
  });
});
